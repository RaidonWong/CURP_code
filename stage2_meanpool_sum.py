"""
Stage 2 (mean-pool stable + ADDITIVE fusion variant) -- rebuttal ablation.

Rebuttal reviewer question (CURP): the stable/dynamic decomposition is a
reasonable story, but is the *structured* two-branch decomposition itself
necessary, or would a single merged user vector do just as well?

This variant collapses the whole user representation into ONE vector:
  Stable branch : mean-pool over PQ-quantized profile embeddings -> 1 token
  Dynamic branch: Contriever query cross-attention              -> 1 token
  Fusion        : element-wise SUM -> 1 combined token (NOT concatenation)
  MLP projects that single token into LLM space, injected via ONE placeholder.

Everything else (frozen LLM, frozen PQ codebook, frozen Contriever, only the
dynamic cross-attn + MLP are trained) is identical to stage2_meanpool.py so the
comparison isolates exactly the fusion choice (structured concat vs single sum).
"""
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from accelerate import Accelerator
from tqdm import tqdm
import argparse

from stage2 import (
    PQCodebookModel, DynamicCrossAttention, MLPProjection,
    mean_pooling, ensure_chat_template, collate_fn, Stage2Dataset,
    tokenize_training_example, left_pad_tokenized_batch,
    inject_placeholder_embeddings, _checkpoint_metadata,
    _safe_unwrap, _state_dict_to_cpu, validate_encoder_dimension,
    create_accelerator,
)


COMBINED_TOKEN = "<USR_EMB>"


def _build_user_prompt_combined(task, combined_str, prompt_text):
    """Single-line, single-placeholder prompt: one unified user vector.

    Mirrors the wording of stage2._build_user_prompt but merges the stable and
    dynamic lines into one, referencing the single combined placeholder.
    """
    if task == 'lamp2':
        line   = (f"A unified personalization vector for this user, combining their stable "
                  f"writing preferences with the aspects of their history most relevant to "
                  f"the current task, is encoded in: {combined_str}")
        suffix = "Output only the answer."
    elif task == 'lamp4':
        line   = (f"A unified personalization vector for this author, combining their stable "
                  f"headline-writing style with the aspects of their past articles most "
                  f"relevant to the current one, is encoded in: {combined_str}")
        suffix = "Generate a headline in this author's personal writing style. Output only the headline."
    elif task == 'lamp3':
        line   = (f"A unified personalization vector for this reviewer, combining their rating "
                  f"calibration with their past reviews most relevant to the current one, is "
                  f"encoded in: {combined_str}")
        suffix = "Based on this user's personal rating behavior, predict the score they assigned. Output only 1, 2, 3, 4, or 5."
    elif task == 'longlamp':
        line   = (f"A unified personalization vector for this reviewer, combining their writing "
                  f"style with the aspects of their reviewing history most relevant to this "
                  f"product, is encoded in: {combined_str}")
        suffix = "Write the review in this reviewer's personal writing style. Output only the review text."
    else:
        line   = (f"A unified personalization vector for this user, combining their stable "
                  f"writing style and preferences with the aspects of their history most "
                  f"relevant to the current task, is encoded in: {combined_str}")
        suffix = "Complete the task above in this user's personal writing style. Output only the answer."
    return f"{line}\n{prompt_text}\n{suffix}"


# ---------------------------------------------------------------------------
# Trainer (mean-pool stable + additive fusion)
# ---------------------------------------------------------------------------

class Stage2MeanPoolSumTrainer:
    def __init__(self, llm_model, mlp_model, dynamic_cross_attn,
                 pq_codebook_model, contriever_model, contriever_tokenizer,
                 tokenizer, accelerator, max_length=2048, task=None):
        self.llm_model            = llm_model
        self.mlp_model            = mlp_model
        self.dynamic_cross_attn   = dynamic_cross_attn
        self.pq_codebook_model    = pq_codebook_model
        self.contriever_model     = contriever_model
        self.contriever_tokenizer = contriever_tokenizer
        self.tokenizer            = tokenizer
        self.accelerator          = accelerator
        self.max_length           = max_length
        self.task                 = task
        self.combined_token       = COMBINED_TOKEN

        for param in self.llm_model.parameters():
            param.requires_grad = False
        for param in self.pq_codebook_model.parameters():
            param.requires_grad = False
        for param in self.contriever_model.parameters():
            param.requires_grad = False
        self.llm_model.eval()
        self.contriever_model.eval()

    def _encode_queries(self, queries, device):
        """Encode query strings with frozen Contriever -> (batch, 1, 768)."""
        encoded = self.contriever_tokenizer(
            queries, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs   = self.contriever_model(**encoded)
            query_emb = mean_pooling(outputs, encoded["attention_mask"])
        return query_emb.unsqueeze(1)   # (batch, 1, 768)

    def compute_loss(self, batch):
        prompts            = batch["prompt"]
        chosens            = batch["chosen"]
        profile_embeddings = batch["profile_embeddings"]   # (B, max_N, emb_dim)
        profile_mask       = batch["profile_mask"]         # (B, max_N), True=padding

        device     = profile_embeddings.device
        batch_size = len(prompts)

        # -- 1. PQ quantize --------------------------------------------------
        with torch.no_grad():
            quantized_embs, _ = self.pq_codebook_model(profile_embeddings)  # (B, max_N, emb_dim)

        # -- 2. Stable branch: mean pool over valid positions ----------------
        valid_mask = (~profile_mask).float().to(device).unsqueeze(-1)  # (B, max_N, 1)
        stable_token = (quantized_embs * valid_mask).sum(1) / valid_mask.sum(1).clamp(min=1e-9)
        stable_token = stable_token.unsqueeze(1)  # (B, 1, emb_dim)

        # -- 3. Dynamic branch -----------------------------------------------
        query_embs    = self._encode_queries(prompts, device)          # (B, 1, emb_dim)
        if query_embs.shape[-1] != quantized_embs.shape[-1]:
            raise ValueError(
                f"Query encoder dim={query_embs.shape[-1]} does not match profile/codebook "
                f"dim={quantized_embs.shape[-1]}"
            )
        query_embs    = query_embs.to(dtype=quantized_embs.dtype)
        dynamic_token = self.dynamic_cross_attn(
            query_embs, quantized_embs,
            key_padding_mask=profile_mask.to(device)
        )   # (B, 1, emb_dim)

        # -- 4. ADDITIVE fusion -> single token, then MLP -> LLM space --------
        combined_token = stable_token + dynamic_token                 # (B, 1, emb_dim)
        llm_embs       = self.mlp_model(combined_token)               # (B, 1, llm_dim)

        # -- 5. Build tokenized prompts (single placeholder) -----------------
        combined_id = self.tokenizer.convert_tokens_to_ids(self.combined_token)

        input_ids_list, labels_list, ph_positions_list = [], [], []

        for i in range(batch_size):
            user_prompt_text = _build_user_prompt_combined(
                self.task, self.combined_token, prompts[i]
            )
            input_ids, labels, ph_positions = tokenize_training_example(
                self.tokenizer,
                user_prompt_text,
                chosens[i],
                self.max_length,
                stable_id=combined_id,
                dynamic_id=-1,
                ablation="stable_only",
                num_query_tokens=1,
                sample_label=f"batch item {i}",
            )

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            ph_positions_list.append(ph_positions)

        input_ids, labels, attention_mask, adjusted_ph = left_pad_tokenized_batch(
            input_ids_list,
            labels_list,
            ph_positions_list,
            self.tokenizer.pad_token_id,
            device,
        )
        input_embs = inject_placeholder_embeddings(
            self.llm_model, input_ids, llm_embs, adjusted_ph
        )

        # -- 8. Forward pass -------------------------------------------------
        outputs = self.llm_model(
            inputs_embeds=input_embs,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(args):
    accelerator = create_accelerator(args)

    if accelerator.is_local_main_process:
        print("Stage 2 (mean-pool stable + ADDITIVE fusion): single combined user token")

    # -- LLM ----------------------------------------------------------------
    print(f"Loading LLM from {args.llm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    ensure_chat_template(tokenizer, args.llm_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Only <USR_EMB> is actually used (single combined token); <USR_STABLE>/<USR_DYN>
    # are added solely to satisfy Stage2Dataset's vocab validation (they never appear
    # in the prompt, so they are never injected).
    new_tokens = [COMBINED_TOKEN, "<USR_STABLE>", "<USR_DYN>"]
    added = tokenizer.add_tokens(new_tokens)
    print(f"Added {added} placeholder token(s): {new_tokens} (only {COMBINED_TOKEN} is used)")

    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    llm_model.resize_token_embeddings(len(tokenizer))
    llm_model.eval()
    llm_dim = llm_model.config.hidden_size
    print(f"LLM hidden size: {llm_dim}")

    # -- Contriever ---------------------------------------------------------
    print(f"Loading Contriever from {args.encoder_path}...")
    os.environ["TRANSFORMERS_NO_FAST_TOKENIZER"] = "1"
    contriever_tokenizer = AutoTokenizer.from_pretrained(
        args.encoder_path, local_files_only=True
    )
    contriever_model = AutoModel.from_pretrained(args.encoder_path)
    contriever_model.eval()
    for param in contriever_model.parameters():
        param.requires_grad = False

    # -- PQ codebook --------------------------------------------------------
    print(f"Loading PQ codebook from {args.codebook_path}...")
    pq_codebook_model = PQCodebookModel(args.codebook_path, device="cpu")
    for param in pq_codebook_model.parameters():
        param.requires_grad = False

    emb_dim = pq_codebook_model.emb_dim
    validate_encoder_dimension(contriever_model, emb_dim, "Contriever")
    print(f"PQ codebook emb_dim: {emb_dim}")

    # -- Dynamic cross-attention --------------------------------------------
    dynamic_cross_attn = DynamicCrossAttention(emb_dim=emb_dim, num_heads=args.num_heads)
    print(f"DynamicCrossAttention: emb_dim={emb_dim}, num_heads={args.num_heads}")

    # -- MLP projection -----------------------------------------------------
    hidden_dim = args.hidden_dim if args.hidden_dim else llm_dim
    mlp_model  = MLPProjection(input_dim=emb_dim, hidden_dim=hidden_dim, output_dim=llm_dim)

    # -- Dataset + DataLoader -----------------------------------------------
    dataset = Stage2Dataset(
        jsonl_path=args.data_path,
        embeddings_path=args.embeddings_path,
        tokenizer=tokenizer,
        max_profile_len=args.max_profile_len,
        max_length=args.max_length,
        emb_dim=emb_dim,
        allow_embedding_dim_mismatch=args.allow_embedding_dim_mismatch,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    # -- Trainer ------------------------------------------------------------
    trainer = Stage2MeanPoolSumTrainer(
        llm_model=llm_model,
        mlp_model=mlp_model,
        dynamic_cross_attn=dynamic_cross_attn,
        pq_codebook_model=pq_codebook_model,
        contriever_model=contriever_model,
        contriever_tokenizer=contriever_tokenizer,
        tokenizer=tokenizer,
        accelerator=accelerator,
        max_length=args.max_length,
        task=args.task,
    )

    # -- Optimizer: only trained components ---------------------------------
    trained_params = (
        list(dynamic_cross_attn.parameters()) +
        list(mlp_model.parameters())
    )
    optimizer = torch.optim.AdamW(
        trained_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # -- Accelerate prepare -------------------------------------------------
    dynamic_cross_attn, mlp_model, optimizer, dataloader = accelerator.prepare(
        dynamic_cross_attn, mlp_model, optimizer, dataloader
    )
    trainer.dynamic_cross_attn = dynamic_cross_attn
    trainer.mlp_model          = mlp_model

    pq_codebook_model = pq_codebook_model.to(accelerator.device)
    contriever_model  = contriever_model.to(accelerator.device)
    trainer.pq_codebook_model = pq_codebook_model
    trainer.contriever_model  = contriever_model

    if not hasattr(llm_model, 'hf_device_map'):
        llm_model = llm_model.to(accelerator.device)
    trainer.llm_model = llm_model

    # -- Training loop ------------------------------------------------------
    accumulation_models = [dynamic_cross_attn, mlp_model]
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
                    parameters = [p for m in accumulation_models for p in m.parameters() if p.requires_grad]
                    accelerator.clip_grad_norm_(parameters, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                global_step += 1

            if accelerator.is_local_main_process:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "update": global_step})

            if accelerator.sync_gradients and args.save_steps > 0 and global_step % args.save_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    _save_checkpoint(accelerator, dynamic_cross_attn, mlp_model,
                                     args, global_step, emb_dim=emb_dim,
                                     optimizer=optimizer)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            _save_checkpoint(accelerator, dynamic_cross_attn, mlp_model,
                             args, step=global_step, epoch=epoch + 1, emb_dim=emb_dim,
                             optimizer=optimizer)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_checkpoint(accelerator, dynamic_cross_attn, mlp_model,
                         args, step=None, final=True, emb_dim=emb_dim,
                         optimizer=optimizer)


def _save_checkpoint(accelerator, dynamic_cross_attn, mlp_model,
                     args, step, final=False, epoch=None, emb_dim=768,
                     optimizer=None):
    if final:
        tag = "stage2_final"
    elif epoch is not None:
        tag = f"stage2_epoch-{epoch}"
    else:
        tag = f"stage2_checkpoint-{step}"
    path = os.path.join(args.output_dir, tag)
    os.makedirs(path, exist_ok=True)

    unwrapped_mlp = _safe_unwrap(mlp_model)

    ckpt = {
        **_checkpoint_metadata(args, step, epoch),
        "stable_type":        "meanpool_sum",
        "combined_token":     COMBINED_TOKEN,
        "dynamic_cross_attn": _state_dict_to_cpu(dynamic_cross_attn),
        "mlp":                _state_dict_to_cpu(mlp_model),
        "input_dim":          unwrapped_mlp.mlp[0].in_features,
        "hidden_dim":         unwrapped_mlp.mlp[0].out_features,
        "output_dim":         unwrapped_mlp.mlp[-1].out_features,
        "num_heads":          args.num_heads,
        "emb_dim":            emb_dim,
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    torch.save(ckpt, os.path.join(path, "model.pt"))
    print(f"Saved {'final' if final else 'checkpoint'} to {path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2 (mean-pool stable + additive fusion): single combined user token"
    )
    parser.add_argument("--data_path",       required=True)
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--llm_path",        required=True)
    parser.add_argument("--codebook_path",   required=True)
    parser.add_argument("--encoder_path",    required=True)
    parser.add_argument("--output_dir",      required=True)

    parser.add_argument("--num_heads",        type=int, default=8)
    parser.add_argument("--hidden_dim",       type=int, default=None)
    parser.add_argument("--max_profile_len",  type=int, default=None)
    parser.add_argument("--max_length",       type=int, default=1024)

    parser.add_argument("--batch_size",                  type=int,   default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=8)
    parser.add_argument("--num_epochs",                  type=int,   default=3)
    parser.add_argument("--learning_rate",               type=float, default=1e-4)
    parser.add_argument("--weight_decay",                type=float, default=0.01)
    parser.add_argument("--max_grad_norm",               type=float, default=1.0)
    parser.add_argument("--save_steps",                  type=int,   default=500)
    parser.add_argument("--seed",                        type=int,   default=42)
    parser.add_argument("--task",                        type=str,   default=None)
    parser.add_argument("--allow_embedding_dim_mismatch", action="store_true")

    args = parser.parse_args()
    train(args)
