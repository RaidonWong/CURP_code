"""
Stage 2 with RoBERTa-based profile embeddings and RoBERTa query encoder.
Differences from stage2.py:
  - Contriever replaced by RoBERTa-large for dynamic query encoding
  - Profile embeddings are 1024-dim (RoBERTa) instead of 768-dim (Contriever)
  - PQ codebook is 1024-dim
  - Q-Former and DCA operate in 1024-dim space
"""
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from accelerate import Accelerator
from tqdm import tqdm
import argparse

from stage2 import (
    PQCodebookModel, QFormer, DynamicCrossAttention, MLPProjection,
    ensure_chat_template, Stage2Dataset, Stage2Trainer, collate_fn, mean_pooling,
    validate_encoder_dimension,
    create_accelerator, _checkpoint_metadata, _safe_unwrap, _state_dict_to_cpu,
)


class Stage2RobertaTrainer(Stage2Trainer):
    """
    Same dual-branch architecture as Stage2Trainer but uses RoBERTa
    for dynamic query encoding instead of Contriever.
    """
    def __init__(self, llm_model, mlp_model, qformer, dynamic_cross_attn,
                 pq_codebook_model, roberta_model, roberta_tokenizer,
                 tokenizer, accelerator, num_query_tokens=4, max_length=2048,
                 task=None, ablation='both'):
        super().__init__(
            llm_model=llm_model,
            mlp_model=mlp_model,
            qformer=qformer,
            dynamic_cross_attn=dynamic_cross_attn,
            pq_codebook_model=pq_codebook_model,
            contriever_model=roberta_model,
            contriever_tokenizer=roberta_tokenizer,
            tokenizer=tokenizer,
            accelerator=accelerator,
            num_query_tokens=num_query_tokens,
            max_length=max_length,
            task=task,
            ablation=ablation,
        )
        self.roberta_model      = roberta_model
        self.roberta_tokenizer  = roberta_tokenizer

    def _encode_queries(self, queries, device):
        """Encode query strings with frozen RoBERTa → (batch, 1, emb_dim)."""
        encoded = self.roberta_tokenizer(
            queries, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs   = self.roberta_model(**encoded)
            query_emb = mean_pooling(outputs, encoded["attention_mask"])
        return query_emb.unsqueeze(1)   # (batch, 1, emb_dim)


def train(args):
    accelerator = create_accelerator(args)

    if accelerator.is_local_main_process:
        print(f"Stage 2 (RoBERTa): Q-Former stable + RoBERTa dynamic query encoder")

    # ── LLM ──────────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    ensure_chat_template(tokenizer, args.llm_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    added = tokenizer.add_tokens(["<USR_STABLE>", "<USR_DYN>"])
    print(f"Added {added} placeholder token(s)")

    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=None
    )
    llm_model.resize_token_embeddings(len(tokenizer))
    llm_model.eval()
    llm_dim = llm_model.config.hidden_size
    print(f"LLM hidden size: {llm_dim}")

    # ── RoBERTa (frozen, for both profile embeddings and query encoding) ──────
    print(f"Loading RoBERTa from {args.encoder_path}...")
    roberta_tokenizer = AutoTokenizer.from_pretrained(args.encoder_path, local_files_only=True)
    roberta_model     = AutoModel.from_pretrained(args.encoder_path, local_files_only=True)
    roberta_model.eval()
    for param in roberta_model.parameters():
        param.requires_grad = False
    print("RoBERTa loaded and frozen.")

    # ── PQ codebook (frozen, 1024-dim) ───────────────────────────────────────
    print(f"Loading PQ codebook from {args.codebook_path}...")
    pq_codebook_model = PQCodebookModel(args.codebook_path, device="cpu")
    for param in pq_codebook_model.parameters():
        param.requires_grad = False
    emb_dim = pq_codebook_model.emb_dim
    validate_encoder_dimension(roberta_model, emb_dim, "RoBERTa")
    print(f"PQ codebook emb_dim: {emb_dim}")

    # ── Q-Former ──────────────────────────────────────────────────────────────
    qformer = QFormer(
        num_query_tokens=args.num_query_tokens,
        emb_dim=emb_dim,
        num_heads=args.num_heads,
        num_layers=args.qformer_layers
    )

    # ── Dynamic cross-attention ───────────────────────────────────────────────
    dynamic_cross_attn = DynamicCrossAttention(emb_dim=emb_dim, num_heads=args.num_heads)

    # ── MLP ───────────────────────────────────────────────────────────────────
    hidden_dim = args.hidden_dim if args.hidden_dim else llm_dim
    mlp_model  = MLPProjection(input_dim=emb_dim, hidden_dim=hidden_dim, output_dim=llm_dim)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = Stage2Dataset(
        jsonl_path=args.data_path,
        embeddings_path=args.embeddings_path,
        tokenizer=tokenizer,
        max_profile_len=args.max_profile_len,
        max_length=args.max_length,
        emb_dim=emb_dim,
        allow_embedding_dim_mismatch=args.allow_embedding_dim_mismatch,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, collate_fn=collate_fn)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Stage2RobertaTrainer(
        llm_model=llm_model, mlp_model=mlp_model,
        qformer=qformer, dynamic_cross_attn=dynamic_cross_attn,
        pq_codebook_model=pq_codebook_model,
        roberta_model=roberta_model, roberta_tokenizer=roberta_tokenizer,
        tokenizer=tokenizer, accelerator=accelerator,
        num_query_tokens=args.num_query_tokens,
        max_length=args.max_length, task=args.task, ablation=args.ablation,
    )

    trained_params = (
        list(qformer.parameters()) +
        list(dynamic_cross_attn.parameters()) +
        list(mlp_model.parameters())
    )
    optimizer = torch.optim.AdamW(trained_params, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    qformer, dynamic_cross_attn, mlp_model, optimizer, dataloader = accelerator.prepare(
        qformer, dynamic_cross_attn, mlp_model, optimizer, dataloader
    )
    trainer.qformer            = qformer
    trainer.dynamic_cross_attn = dynamic_cross_attn
    trainer.mlp_model          = mlp_model

    pq_codebook_model = pq_codebook_model.to(accelerator.device)
    roberta_model     = roberta_model.to(accelerator.device)
    trainer.pq_codebook_model = pq_codebook_model
    trainer.roberta_model     = roberta_model

    if not hasattr(llm_model, 'hf_device_map'):
        llm_model = llm_model.to(accelerator.device)
    trainer.llm_model = llm_model

    # ── Training loop ─────────────────────────────────────────────────────────
    accumulation_models = [qformer, dynamic_cross_attn, mlp_model]
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    for epoch in range(args.num_epochs):
        if accelerator.is_local_main_process:
            print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            with accelerator.accumulate(*accumulation_models):
                loss = trainer.compute_loss(batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    trainable_parameters = [
                        parameter
                        for model in accumulation_models
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ]
                    accelerator.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                global_step += 1
            if accelerator.is_local_main_process:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "update": global_step})
            if accelerator.sync_gradients and args.save_steps > 0 and global_step % args.save_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    _save_checkpoint(
                        accelerator, qformer, dynamic_cross_attn, mlp_model,
                        args, global_step, emb_dim=emb_dim, optimizer=optimizer,
                    )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            _save_checkpoint(accelerator, qformer, dynamic_cross_attn, mlp_model,
                             args, global_step, epoch=epoch + 1, emb_dim=emb_dim,
                             optimizer=optimizer)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_checkpoint(accelerator, qformer, dynamic_cross_attn, mlp_model,
                         args, step=None, final=True, emb_dim=emb_dim,
                         optimizer=optimizer)


def _save_checkpoint(accelerator, qformer, dynamic_cross_attn, mlp_model,
                     args, step, final=False, epoch=None, emb_dim=1024,
                     optimizer=None, encoder_type="roberta"):
    tag  = "stage2_final" if final else (f"stage2_epoch-{epoch}" if epoch else f"stage2_checkpoint-{step}")
    path = os.path.join(args.output_dir, tag)
    os.makedirs(path, exist_ok=True)

    uq = _safe_unwrap(qformer)
    um = _safe_unwrap(mlp_model)

    checkpoint = {
        **_checkpoint_metadata(args, step, epoch),
        "encoder_type":       encoder_type,
        "qformer":            _state_dict_to_cpu(qformer),
        "num_query_tokens":   args.num_query_tokens,
        "qformer_layers":     args.qformer_layers,
        "num_heads":          args.num_heads,
        "dynamic_cross_attn": _state_dict_to_cpu(dynamic_cross_attn),
        "mlp":                _state_dict_to_cpu(mlp_model),
        "input_dim":          um.mlp[0].in_features,
        "hidden_dim":         um.mlp[0].out_features,
        "output_dim":         um.mlp[-1].out_features,
        "emb_dim":            uq.query_tokens.shape[-1],
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    torch.save(checkpoint, os.path.join(path, "model.pt"))
    print(f"Saved {'final' if final else 'checkpoint'} to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2 with RoBERTa profile embeddings and query encoder"
    )
    parser.add_argument("--data_path",       required=True)
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--llm_path",        required=True)
    parser.add_argument("--codebook_path",   required=True)
    parser.add_argument("--encoder_path",    required=True, help="RoBERTa model dir")
    parser.add_argument("--output_dir",      required=True)

    parser.add_argument("--num_query_tokens", type=int, default=4)
    parser.add_argument("--qformer_layers",   type=int, default=2)
    parser.add_argument("--num_heads",        type=int, default=8)
    parser.add_argument("--hidden_dim",       type=int, default=None)
    parser.add_argument("--max_profile_len",  type=int, default=None)
    parser.add_argument("--max_length",       type=int, default=1024)
    parser.add_argument("--ablation",         type=str, default='both',
                        choices=['both', 'stable_only', 'dynamic_only'])

    parser.add_argument("--batch_size",                  type=int,   default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=8)
    parser.add_argument("--num_epochs",                  type=int,   default=3)
    parser.add_argument("--learning_rate",               type=float, default=1e-4)
    parser.add_argument("--weight_decay",                type=float, default=0.01)
    parser.add_argument("--max_grad_norm",               type=float, default=1.0)
    parser.add_argument("--save_steps",                  type=int,   default=99999)
    parser.add_argument("--seed",                        type=int,   default=42)
    parser.add_argument("--task",                        type=str,   default=None)
    parser.add_argument("--allow_embedding_dim_mismatch", action="store_true")

    args = parser.parse_args()
    train(args)
