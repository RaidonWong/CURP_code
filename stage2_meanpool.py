"""CURP ablation with mean pooling for the stable branch."""

import argparse
import os

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from stage2 import (
    DynamicCrossAttention,
    MLPProjection,
    MeanPoolStable,
    PQCodebookModel,
    Stage2Dataset,
    Stage2Trainer,
    _save_checkpoint,
    collate_fn,
    create_accelerator,
    ensure_chat_template,
    validate_encoder_dimension,
)


def train(args):
    args.num_query_tokens = 1
    args.qformer_layers = 0
    args.stable_pooling = "mean"
    args.finetune_codebook = False
    accelerator = create_accelerator(args)

    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    ensure_chat_template(tokenizer, args.llm_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(["<USR_STABLE>", "<USR_DYN>"])

    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    llm_model.resize_token_embeddings(len(tokenizer))
    llm_model.eval()
    llm_dim = llm_model.config.hidden_size

    os.environ["TRANSFORMERS_NO_FAST_TOKENIZER"] = "1"
    encoder_tokenizer = AutoTokenizer.from_pretrained(
        args.encoder_path, local_files_only=True
    )
    encoder_model = AutoModel.from_pretrained(
        args.encoder_path, local_files_only=True
    ).eval()
    for parameter in encoder_model.parameters():
        parameter.requires_grad = False

    codebook = PQCodebookModel(args.codebook_path, device="cpu")
    emb_dim = codebook.emb_dim
    validate_encoder_dimension(encoder_model, emb_dim, "Contriever")
    stable_pool = MeanPoolStable(emb_dim)
    dynamic_cross_attn = DynamicCrossAttention(
        emb_dim=emb_dim, num_heads=args.num_heads
    )
    hidden_dim = args.hidden_dim or llm_dim
    mlp_model = MLPProjection(
        input_dim=emb_dim, hidden_dim=hidden_dim, output_dim=llm_dim
    )

    dataset = Stage2Dataset(
        args.data_path,
        args.embeddings_path,
        tokenizer,
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
        collate_fn=collate_fn,
    )
    trainer = Stage2Trainer(
        llm_model,
        mlp_model,
        stable_pool,
        dynamic_cross_attn,
        codebook,
        encoder_model,
        encoder_tokenizer,
        tokenizer,
        accelerator,
        num_query_tokens=1,
        max_length=args.max_length,
        task=args.task,
        ablation="both",
    )

    trainable_modules = [dynamic_cross_attn, mlp_model]
    optimizer = torch.optim.AdamW(
        [parameter for module in trainable_modules for parameter in module.parameters()],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    dynamic_cross_attn, mlp_model, optimizer, dataloader = accelerator.prepare(
        dynamic_cross_attn, mlp_model, optimizer, dataloader
    )
    trainable_modules = [dynamic_cross_attn, mlp_model]
    trainer.qformer = stable_pool.to(accelerator.device)
    trainer.dynamic_cross_attn = dynamic_cross_attn
    trainer.mlp_model = mlp_model
    trainer.pq_codebook_model = codebook.to(accelerator.device)
    trainer.contriever_model = encoder_model.to(accelerator.device)
    trainer.llm_model = llm_model.to(accelerator.device)

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            with accelerator.accumulate(*trainable_modules):
                loss = trainer.compute_loss(batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    parameters = [
                        parameter
                        for module in trainable_modules
                        for parameter in module.parameters()
                        if parameter.requires_grad
                    ]
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
                    _save_checkpoint(
                        accelerator, trainer.qformer, dynamic_cross_attn, mlp_model,
                        args, global_step, optimizer=optimizer,
                    )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            _save_checkpoint(
                accelerator, trainer.qformer, dynamic_cross_attn, mlp_model,
                args, global_step, epoch=epoch + 1, optimizer=optimizer,
            )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_checkpoint(
            accelerator, trainer.qformer, dynamic_cross_attn, mlp_model,
            args, step=None, final=True, optimizer=optimizer,
        )


def build_parser():
    parser = argparse.ArgumentParser(description="CURP mean-pool stable ablation")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--llm_path", required=True)
    parser.add_argument("--codebook_path", required=True)
    parser.add_argument("--encoder_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--max_profile_len", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=99999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default=None)
    parser.add_argument("--allow_embedding_dim_mismatch", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
