"""Inference for mean-pool stable + dynamic model with ADDITIVE fusion.

Companion to stage2_meanpool_sum.py: the user representation is a single vector
(mean-pool stable + dynamic, summed), injected via one <USR_EMB> placeholder.
"""
import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse

from stage2 import (
    PQCodebookModel, DynamicCrossAttention, MLPProjection,
    mean_pooling, ensure_chat_template, load_embedding_tensor,
    tokenize_generation_prompt, left_pad_tokenized_batch,
    inject_placeholder_embeddings, validate_encoder_dimension,
)
from stage2_meanpool_sum import _build_user_prompt_combined, COMBINED_TOKEN
from infer import InferDataset, infer_collate


@torch.no_grad()
def generate_batch(batch, llm_model, dynamic_cross_attn, mlp_model,
                   pq_codebook_model, contriever_model, contriever_tokenizer,
                   tokenizer, max_new_tokens, device, task=None, max_length=2048):
    prompts            = batch['prompts']
    profile_embeddings = batch['profile_embeddings'].to(device)
    profile_mask       = batch['profile_mask'].to(device)
    B                  = len(prompts)

    combined_id = tokenizer.convert_tokens_to_ids(COMBINED_TOKEN)

    # 1. PQ quantize
    quantized_embs, _ = pq_codebook_model(profile_embeddings)

    # 2. Stable branch: mean pool
    valid_mask   = (~profile_mask).float().unsqueeze(-1)          # (B, max_N, 1)
    stable_token = (quantized_embs * valid_mask).sum(1) / valid_mask.sum(1).clamp(min=1e-9)
    stable_token = stable_token.unsqueeze(1)                      # (B, 1, emb_dim)

    # 3. Dynamic branch
    enc = contriever_tokenizer(
        prompts, padding=True, truncation=True, max_length=512, return_tensors='pt'
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = contriever_model(**enc)
    query_emb = mean_pooling(out, enc['attention_mask']).unsqueeze(1)  # (B, 1, 768)
    if query_emb.shape[-1] != quantized_embs.shape[-1]:
        raise ValueError(
            f"Query encoder dim={query_emb.shape[-1]} does not match profile dim={quantized_embs.shape[-1]}"
        )
    query_emb = query_emb.to(dtype=quantized_embs.dtype)
    dynamic_tok = dynamic_cross_attn(query_emb, quantized_embs, key_padding_mask=profile_mask)

    # 4. ADDITIVE fusion -> single token, MLP -> LLM space
    combined_tok = stable_token + dynamic_tok                    # (B, 1, emb_dim)
    llm_embs     = mlp_model(combined_tok)                       # (B, 1, llm_dim)

    # 5. Build prompt strings (single placeholder)
    input_ids_list, ph_positions_list = [], []
    for i in range(B):
        user_text  = _build_user_prompt_combined(task, COMBINED_TOKEN, prompts[i])
        ids, ph_pos = tokenize_generation_prompt(
            tokenizer,
            user_text,
            max_length,
            stable_id=combined_id,
            dynamic_id=-1,
            ablation='stable_only',
            num_query_tokens=1,
            sample_label=f'inference batch item {i}',
        )
        input_ids_list.append(ids)
        ph_positions_list.append(ph_pos)

    input_ids, _, attention_mask, adjusted_positions = left_pad_tokenized_batch(
        input_ids_list, None, ph_positions_list, tokenizer.pad_token_id, device
    )
    input_embs = inject_placeholder_embeddings(
        llm_model, input_ids, llm_embs, adjusted_positions
    )

    # 8. Generate
    outputs = llm_model.generate(
        inputs_embeds=input_embs,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt    = torch.load(args.checkpoint_path, map_location='cpu')
    emb_dim = ckpt.get('emb_dim', 768)
    num_heads   = ckpt.get('num_heads', 8)
    input_dim   = ckpt.get('input_dim', emb_dim)
    hidden_dim  = ckpt.get('hidden_dim', None)
    output_dim  = ckpt.get('output_dim', 3584)
    task = args.task if args.task is not None else ckpt.get('task')
    max_length = args.max_length if args.max_length is not None else ckpt.get('max_length', 2048)
    max_profile_len = (
        args.max_profile_len
        if args.max_profile_len is not None
        else ckpt.get('max_profile_len')
    )

    # LLM
    print(f'Loading LLM from {args.llm_path}...')
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    ensure_chat_template(tokenizer, args.llm_path)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Match training vocab exactly (same tokens, same order) so <USR_EMB> keeps the
    # same id; the extra two are unused padding tokens.
    tokenizer.add_tokens([COMBINED_TOKEN, "<USR_STABLE>", "<USR_DYN>"])

    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=None
    )
    llm_model.resize_token_embeddings(len(tokenizer))
    llm_model = llm_model.to(device).eval()
    if output_dim != llm_model.config.hidden_size:
        raise ValueError(
            f"Checkpoint output_dim={output_dim} does not match LLM hidden_size={llm_model.config.hidden_size}"
        )
    model_context = getattr(llm_model.config, 'max_position_embeddings', None)
    if model_context is not None:
        max_length = min(max_length, int(model_context) - args.max_new_tokens)
    if max_length <= 0:
        raise ValueError('No prompt budget remains after reserving max_new_tokens')

    # Contriever
    print(f'Loading Contriever from {args.encoder_path}...')
    os.environ['TRANSFORMERS_NO_FAST_TOKENIZER'] = '1'
    contriever_tokenizer = AutoTokenizer.from_pretrained(
        args.encoder_path, local_files_only=True
    )
    contriever_model = AutoModel.from_pretrained(args.encoder_path).to(device).eval()

    # PQ codebook
    print(f'Loading PQ codebook from {args.codebook_path}...')
    pq_model = PQCodebookModel(args.codebook_path, device=str(device)).eval().to(device)
    validate_encoder_dimension(contriever_model, emb_dim, 'Contriever')

    # Stage 2 modules
    dca = DynamicCrossAttention(emb_dim=emb_dim, num_heads=num_heads)
    dca.load_state_dict(ckpt['dynamic_cross_attn'])
    dca = dca.to(device).eval()

    mlp = MLPProjection(input_dim=input_dim,
                        hidden_dim=hidden_dim if hidden_dim else output_dim,
                        output_dim=output_dim)
    mlp.load_state_dict(ckpt['mlp'])
    mlp = mlp.to(device).eval()
    print('Stage2 modules loaded.')

    # Load embeddings tensor (fixes the path-vs-tensor bug in infer_meanpool.py)
    embeddings = load_embedding_tensor(
        args.embeddings_path,
        expected_dim=emb_dim,
        allow_dim_mismatch=args.allow_embedding_dim_mismatch,
    )

    dataset = InferDataset(
        args.data_path, embeddings=embeddings, max_profile_len=max_profile_len
    )
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=0, collate_fn=infer_collate)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    num_written = 0
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for batch in tqdm(loader, desc='Generating'):
            preds = generate_batch(
                batch, llm_model, dca, mlp,
                pq_model, contriever_model, contriever_tokenizer,
                tokenizer, args.max_new_tokens, device, task=task,
                max_length=max_length,
            )
            for sid, prompt, ref, pred in zip(
                batch['ids'], batch['prompts'], batch['references'], preds
            ):
                record = {'id': sid, 'input': prompt, 'output': ref,
                          'reference': ref, 'prediction': pred}
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                num_written += 1
            f.flush()
    print(f'Saved {num_written} predictions to {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 2 mean-pool + additive fusion inference')
    parser.add_argument('--data_path',       required=True)
    parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--llm_path',        required=True)
    parser.add_argument('--codebook_path',   required=True)
    parser.add_argument('--encoder_path',    required=True)
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--output_path',     required=True)
    parser.add_argument('--batch_size',      type=int, default=4)
    parser.add_argument('--max_new_tokens',  type=int, default=128)
    parser.add_argument('--max_length',      type=int, default=None)
    parser.add_argument('--max_profile_len', type=int, default=None)
    parser.add_argument('--task',            type=str, default=None)
    parser.add_argument('--allow_embedding_dim_mismatch', action='store_true')
    args = parser.parse_args()
    main(args)
