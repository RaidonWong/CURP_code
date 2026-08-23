"""
Stage 2: BGE encoder + dual MLP projectors.
BGE CLS+L2-normalize for dynamic query encoding.
Separate mlp_stable / mlp_dynamic for projection.
"""
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from stage2 import (
    PQCodebookModel, QFormer, DynamicCrossAttention, MLPProjection,
    Stage2Dataset, collate_fn, ensure_chat_template, validate_encoder_dimension,
    create_accelerator,
)
from stage2_dualmlp import Stage2DualMLPTrainer, _save_checkpoint_dual


class Stage2BGEDualMLPTrainer(Stage2DualMLPTrainer):
    """Dual MLP with BGE CLS+L2-normalize for dynamic query encoding."""

    def _encode_queries(self, queries, device):
        encoded = self.roberta_tokenizer(
            queries, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            out = self.roberta_model(**encoded)
            query_emb = F.normalize(out[0][:, 0], p=2, dim=1)
        return query_emb.unsqueeze(1)   # (batch, 1, emb_dim)


def train(args):
    accelerator = create_accelerator(args)

    if accelerator.is_local_main_process:
        print("Stage 2 (BGE + Dual MLP): BGE CLS query encoder, separate stable/dynamic projectors")

    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    ensure_chat_template(tokenizer, args.llm_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(["<USR_STABLE>", "<USR_DYN>"])

    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=None
    )
    llm_model.resize_token_embeddings(len(tokenizer))
    llm_model.eval()
    llm_dim = llm_model.config.hidden_size

    bge_tokenizer = AutoTokenizer.from_pretrained(args.encoder_path, local_files_only=True)
    bge_model     = AutoModel.from_pretrained(args.encoder_path, local_files_only=True)
    bge_model.eval()
    for p in bge_model.parameters():
        p.requires_grad = False

    pq_codebook_model = PQCodebookModel(args.codebook_path, device="cpu")
    for p in pq_codebook_model.parameters():
        p.requires_grad = False
    emb_dim = pq_codebook_model.emb_dim
    validate_encoder_dimension(bge_model, emb_dim, "BGE")

    qformer            = QFormer(num_query_tokens=args.num_query_tokens, emb_dim=emb_dim,
                                 num_heads=args.num_heads, num_layers=args.qformer_layers)
    dynamic_cross_attn = DynamicCrossAttention(emb_dim=emb_dim, num_heads=args.num_heads)

    hidden_dim  = args.hidden_dim if args.hidden_dim else llm_dim
    mlp_stable  = MLPProjection(input_dim=emb_dim, hidden_dim=hidden_dim, output_dim=llm_dim)
    mlp_dynamic = MLPProjection(input_dim=emb_dim, hidden_dim=hidden_dim, output_dim=llm_dim)
    print(f"Dual MLP: stable {emb_dim}->{hidden_dim}->{llm_dim}, "
          f"dynamic {emb_dim}->{hidden_dim}->{llm_dim}")

    dataset = Stage2Dataset(
        jsonl_path=args.data_path, embeddings_path=args.embeddings_path,
        tokenizer=tokenizer, max_profile_len=args.max_profile_len,
        max_length=args.max_length, emb_dim=emb_dim,
        allow_embedding_dim_mismatch=args.allow_embedding_dim_mismatch,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, collate_fn=collate_fn)

    trainer = Stage2BGEDualMLPTrainer(
        llm_model=llm_model, mlp_stable=mlp_stable, mlp_dynamic=mlp_dynamic,
        qformer=qformer, dynamic_cross_attn=dynamic_cross_attn,
        pq_codebook_model=pq_codebook_model,
        contriever_model=bge_model,       # satisfies parent freeze loop
        contriever_tokenizer=bge_tokenizer,
        tokenizer=tokenizer, accelerator=accelerator,
        num_query_tokens=args.num_query_tokens,
        max_length=args.max_length, task=args.task, ablation=args.ablation,
    )
    # _encode_queries uses roberta_model/roberta_tokenizer slots
    trainer.roberta_model     = bge_model
    trainer.roberta_tokenizer = bge_tokenizer

    trained_params = (list(qformer.parameters()) +
                      list(dynamic_cross_attn.parameters()) +
                      list(mlp_stable.parameters()) +
                      list(mlp_dynamic.parameters()))
    optimizer = torch.optim.AdamW(trained_params, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    qformer, dynamic_cross_attn, mlp_stable, mlp_dynamic, optimizer, dataloader = \
        accelerator.prepare(qformer, dynamic_cross_attn, mlp_stable, mlp_dynamic,
                            optimizer, dataloader)
    trainer.qformer            = qformer
    trainer.dynamic_cross_attn = dynamic_cross_attn
    trainer.mlp_stable         = mlp_stable
    trainer.mlp_dynamic        = mlp_dynamic

    pq_codebook_model = pq_codebook_model.to(accelerator.device)
    bge_model         = bge_model.to(accelerator.device)
    trainer.pq_codebook_model = pq_codebook_model
    trainer.roberta_model     = bge_model

    if not hasattr(llm_model, 'hf_device_map'):
        llm_model = llm_model.to(accelerator.device)
    trainer.llm_model = llm_model

    accumulation_models = [qformer, dynamic_cross_attn, mlp_stable, mlp_dynamic]
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    for epoch in range(args.num_epochs):
        if accelerator.is_local_main_process:
            print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            with accelerator.accumulate(*accumulation_models):
                loss = trainer.compute_loss(batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    params = [p for m in accumulation_models for p in m.parameters() if p.requires_grad]
                    accelerator.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                global_step += 1
            if accelerator.is_local_main_process:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "update": global_step})
            if accelerator.sync_gradients and args.save_steps > 0 and global_step % args.save_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    _save_checkpoint_dual(accelerator, qformer, dynamic_cross_attn,
                                          mlp_stable, mlp_dynamic, args, global_step,
                                          optimizer=optimizer, encoder_type="bge")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            _save_checkpoint_dual(accelerator, qformer, dynamic_cross_attn,
                                  mlp_stable, mlp_dynamic, args,
                                  step=global_step, epoch=epoch+1,
                                  optimizer=optimizer, encoder_type="bge")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_checkpoint_dual(accelerator, qformer, dynamic_cross_attn,
                              mlp_stable, mlp_dynamic, args,
                              step=None, final=True, optimizer=optimizer,
                              encoder_type="bge")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",       required=True)
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--llm_path",        required=True)
    parser.add_argument("--codebook_path",   required=True)
    parser.add_argument("--encoder_path",    required=True)
    parser.add_argument("--output_dir",      required=True)
    parser.add_argument("--num_query_tokens", type=int,   default=4)
    parser.add_argument("--qformer_layers",   type=int,   default=2)
    parser.add_argument("--num_heads",        type=int,   default=8)
    parser.add_argument("--hidden_dim",       type=int,   default=None)
    parser.add_argument("--max_profile_len",  type=int,   default=None)
    parser.add_argument("--max_length",       type=int,   default=1024)
    parser.add_argument("--ablation",         type=str,   default='both',
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
