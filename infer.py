import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse

from stage2 import (
    PQCodebookModel, IdentityCodebookModel, QFormer, MeanPoolStable,
    DynamicCrossAttention, MLPProjection,
    mean_pooling, ensure_chat_template, _build_user_prompt,
    STABLE_TOKEN, DYNAMIC_TOKEN, load_embedding_tensor,
    tokenize_generation_prompt, left_pad_tokenized_batch,
    inject_placeholder_embeddings, validate_encoder_dimension,
)
from encode import profile_item_to_text


def encode_profiles_online(data, contriever_model, contriever_tokenizer, device, batch_size=128):
    order_text = {}
    for rec in data:
        for item in rec.get('profile', []):
            order = item.get('order')
            if isinstance(order, int) and order >= 0:
                text = profile_item_to_text(item)
                if text:
                    order_text[order] = text

    hidden_size = int(contriever_model.config.hidden_size)
    if not order_text:
        return torch.zeros(1, hidden_size)

    max_order = max(order_text.keys())
    if max_order > max(1_000_000, len(order_text) * 100):
        raise ValueError(
            f"Profile order space is unexpectedly sparse (max order={max_order}, "
            f"valid items={len(order_text)}); refusing a potentially huge allocation"
        )
    orders = sorted(order_text.keys())
    texts  = [order_text[o] for o in orders]

    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Encoding profiles online'):
        batch = texts[i : i + batch_size]
        enc = contriever_tokenizer(batch, padding=True, truncation=True,
                                   max_length=512, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out  = contriever_model(**enc)
            embs = mean_pooling(out, enc['attention_mask']).cpu()
        all_embs.append(embs)

    all_embs = torch.cat(all_embs, dim=0)
    if all_embs.shape[1] != hidden_size:
        hidden_size = all_embs.shape[1]
    result = torch.zeros(max_order + 1, hidden_size, dtype=all_embs.dtype)
    for idx, order in enumerate(orders):
        result[order] = all_embs[idx]
    print(f'Online encoding: {len(orders)} profile items → shape {result.shape}')
    return result


class InferDataset(Dataset):
    def __init__(self, jsonl_path, embeddings, max_profile_len=None):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

        self.embeddings = embeddings
        self.emb_dim    = embeddings.shape[1]
        self.max_profile_len = max_profile_len
        print(f"Loaded {len(self.data)} samples, embeddings {self.embeddings.shape}")
        invalid_orders = 0
        empty_profiles = 0
        for item in self.data:
            profile = item.get('profile', [])
            if self.max_profile_len is not None:
                profile = profile[:self.max_profile_len]
            valid = 0
            for profile_item in profile:
                order = profile_item.get('order')
                if isinstance(order, int) and 0 <= order < len(self.embeddings):
                    valid += 1
                else:
                    invalid_orders += 1
            if valid == 0:
                empty_profiles += 1
        if invalid_orders:
            print(f"WARNING: skipped {invalid_orders} invalid profile embedding order(s)")
        if empty_profiles:
            print(f"WARNING: {empty_profiles} sample(s) use a zero fallback profile token")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        profile = item.get('profile', [])
        if self.max_profile_len is not None:
            profile = profile[:self.max_profile_len]
        embs = []
        for p in profile:
            order = p.get('order')
            if isinstance(order, int) and 0 <= order < len(self.embeddings):
                embs.append(self.embeddings[order])
        if not embs:
            embs.append(torch.zeros(self.emb_dim, dtype=self.embeddings.dtype))
        return {
            'id': item.get('id', str(idx)),
            'prompt': item.get('input', ''),
            'reference': item.get('output', ''),
            'profile_embeddings': torch.stack(embs),
        }


def infer_collate(batch):
    ids        = [b['id'] for b in batch]
    prompts    = [b['prompt'] for b in batch]
    references = [b['reference'] for b in batch]
    embs_list  = [b['profile_embeddings'] for b in batch]

    max_N   = max(e.size(0) for e in embs_list)
    emb_dim = embs_list[0].size(1)
    B       = len(batch)

    padded = torch.zeros(B, max_N, emb_dim, dtype=embs_list[0].dtype)
    mask   = torch.ones(B, max_N, dtype=torch.bool)
    for i, embs in enumerate(embs_list):
        N = embs.size(0)
        padded[i, :N] = embs
        mask[i, :N]   = False

    return {
        'ids': ids,
        'prompts': prompts,
        'references': references,
        'profile_embeddings': padded,
        'profile_mask': mask,
    }


@torch.no_grad()
def generate_batch(batch, llm_model, qformer, dynamic_cross_attn, mlp_model,
                   pq_codebook_model, contriever_model, contriever_tokenizer,
                   tokenizer, num_query_tokens, max_new_tokens, device, task=None,
                   ablation='both', max_length=2048, mlp_dynamic=None,
                   encoder_type='contriever'):
    prompts            = batch['prompts']
    profile_embeddings = batch['profile_embeddings'].to(device)
    profile_mask       = batch['profile_mask'].to(device)
    B                  = len(prompts)

    stable_token  = STABLE_TOKEN
    dynamic_token = DYNAMIC_TOKEN
    stable_id     = tokenizer.convert_tokens_to_ids(stable_token)
    dynamic_id    = tokenizer.convert_tokens_to_ids(dynamic_token)

    quantized_embs, _ = pq_codebook_model(profile_embeddings)

    if ablation != 'dynamic_only':
        stable_tokens = qformer(quantized_embs, key_padding_mask=profile_mask)

    if ablation != 'stable_only':
        enc = contriever_tokenizer(
            prompts, padding=True, truncation=True, max_length=512, return_tensors='pt'
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = contriever_model(**enc)
        if encoder_type == 'bge':
            query_emb = torch.nn.functional.normalize(out[0][:, 0], p=2, dim=1)
        else:
            query_emb = mean_pooling(out, enc['attention_mask'])
        query_emb = query_emb.unsqueeze(1)
        if query_emb.shape[-1] != quantized_embs.shape[-1]:
            raise ValueError(
                f"Query encoder dimension {query_emb.shape[-1]} does not match "
                f"profile/codebook dimension {quantized_embs.shape[-1]}"
            )
        query_emb = query_emb.to(dtype=quantized_embs.dtype)
        dynamic_tok = dynamic_cross_attn(query_emb, quantized_embs, key_padding_mask=profile_mask)

    if mlp_dynamic is not None:
        if ablation == 'stable_only':
            llm_embs = mlp_model(stable_tokens)
        elif ablation == 'dynamic_only':
            llm_embs = mlp_dynamic(dynamic_tok)
        else:
            llm_embs = torch.cat(
                [mlp_model(stable_tokens), mlp_dynamic(dynamic_tok)], dim=1
            )
    else:
        if ablation == 'stable_only':
            all_tokens = stable_tokens
        elif ablation == 'dynamic_only':
            all_tokens = dynamic_tok
        else:
            all_tokens = torch.cat([stable_tokens, dynamic_tok], dim=1)
        llm_embs = mlp_model(all_tokens)

    input_ids_list = []
    ph_positions_list = []
    for i in range(B):
        stable_str = ' '.join([stable_token] * num_query_tokens)
        user_text = _build_user_prompt(task, stable_str, dynamic_token, prompts[i], ablation=ablation)
        ids, positions = tokenize_generation_prompt(
            tokenizer,
            user_text,
            max_length,
            stable_id,
            dynamic_id,
            ablation,
            num_query_tokens,
            sample_label=f"inference batch item {i}",
        )
        input_ids_list.append(ids)
        ph_positions_list.append(positions)

    input_ids, _, attention_mask, adjusted_positions = left_pad_tokenized_batch(
        input_ids_list,
        None,
        ph_positions_list,
        tokenizer.pad_token_id,
        device,
    )
    input_embs = inject_placeholder_embeddings(
        llm_model, input_ids, llm_embs, adjusted_positions
    )

    outputs = llm_model.generate(
        inputs_embeds=input_embs,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.checkpoint_path, map_location='cpu')
    checkpoint_task = ckpt.get('task')
    checkpoint_ablation = ckpt.get('ablation')
    if (
        args.task is not None
        and checkpoint_task is not None
        and args.task != checkpoint_task
        and not args.allow_checkpoint_config_override
    ):
        raise ValueError(
            f"--task={args.task} conflicts with checkpoint task={checkpoint_task}. "
            "Pass --allow_checkpoint_config_override only if this is intentional."
        )
    if (
        args.ablation is not None
        and checkpoint_ablation is not None
        and args.ablation != checkpoint_ablation
        and not args.allow_checkpoint_config_override
    ):
        raise ValueError(
            f"--ablation={args.ablation} conflicts with checkpoint ablation={checkpoint_ablation}."
        )
    task = args.task if args.task is not None else checkpoint_task
    ablation = args.ablation if args.ablation is not None else ckpt.get('ablation', 'both')
    max_length = args.max_length if args.max_length is not None else ckpt.get('max_length', 2048)
    max_profile_len = (
        args.max_profile_len
        if args.max_profile_len is not None
        else ckpt.get('max_profile_len')
    )
    encoder_type = args.encoder_type or ckpt.get('encoder_type', 'contriever')
    if encoder_type not in {'contriever', 'roberta', 'bge'}:
        raise ValueError(f"Unsupported encoder_type={encoder_type!r}")
    num_query_tokens = ckpt.get('num_query_tokens', 4)
    qformer_layers   = ckpt.get('qformer_layers',   2)
    num_heads        = ckpt.get('num_heads',         8)
    emb_dim          = ckpt.get('emb_dim',           768)
    input_dim        = ckpt.get('input_dim',         emb_dim)
    hidden_dim       = ckpt.get('hidden_dim',        None)
    output_dim       = ckpt.get('output_dim',        3584)

    print(f'Loading LLM from {args.llm_path}...')
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    ensure_chat_template(tokenizer, args.llm_path)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(['<USR_STABLE>', '<USR_DYN>'])

    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=None
    )
    llm_model.resize_token_embeddings(len(tokenizer))
    llm_model = llm_model.to(device).eval()
    if output_dim != llm_model.config.hidden_size:
        raise ValueError(
            f"Checkpoint projector output_dim={output_dim} does not match LLM hidden_size="
            f"{llm_model.config.hidden_size}"
        )
    model_context = getattr(llm_model.config, 'max_position_embeddings', None)
    if model_context is not None:
        max_length = min(max_length, int(model_context) - args.max_new_tokens)
    if max_length <= 0:
        raise ValueError("No prompt budget remains after reserving max_new_tokens")
    print('LLM loaded.')

    print(f'Loading encoder from {args.encoder_path}...')
    contriever_tokenizer = AutoTokenizer.from_pretrained(
        args.encoder_path, local_files_only=True
    )
    contriever_model = AutoModel.from_pretrained(args.encoder_path).to(device).eval()
    print('Contriever loaded.')

    no_pq = bool(ckpt.get('no_pq', False))
    if no_pq:
        pq_model = IdentityCodebookModel(emb_dim).to(device).eval()
        print('Using no-PQ identity profile path from checkpoint metadata.')
    else:
        if not args.codebook_path:
            raise ValueError('--codebook_path is required for a quantized checkpoint')
        print(f'Loading PQ codebook from {args.codebook_path}...')
        pq_model = PQCodebookModel(args.codebook_path, device=str(device)).to(device).eval()
    validate_encoder_dimension(contriever_model, emb_dim, encoder_type)
    # rebuttal: if the stage-2 checkpoint carries a fine-tuned codebook, use it
    if not no_pq and ckpt.get("finetuned_codebook") and "codebooks" in ckpt:
        import torch.nn as _nn
        pq_model.codebooks = _nn.ParameterList([
            _nn.Parameter(cb.to(device), requires_grad=False) for cb in ckpt["codebooks"]
        ])
        if ckpt.get("quant_type", pq_model.quant_type) != pq_model.quant_type:
            raise ValueError(
                f"Fine-tuned checkpoint quant_type={ckpt.get('quant_type')} conflicts with "
                f"base codebook quant_type={pq_model.quant_type}"
            )
        if pq_model.quant_type == "opq" and "rotation" in ckpt:
            pq_model.rotation = _nn.Parameter(
                ckpt["rotation"].to(device), requires_grad=False
            )
        pq_model = pq_model.to(device).eval()
        print(f"[finetune_codebook] loaded fine-tuned codebook from checkpoint ({len(ckpt['codebooks'])} subspaces)")

    if ckpt.get('stable_pooling') == 'mean':
        qformer = MeanPoolStable(emb_dim)
    else:
        qformer = QFormer(num_query_tokens=num_query_tokens, emb_dim=emb_dim,
                          num_heads=num_heads, num_layers=qformer_layers)
    qformer.load_state_dict(ckpt['qformer'])
    qformer = qformer.to(device).eval()

    dca = DynamicCrossAttention(emb_dim=emb_dim, num_heads=num_heads)
    dca.load_state_dict(ckpt['dynamic_cross_attn'])
    dca = dca.to(device).eval()

    mlp = MLPProjection(
        input_dim=input_dim,
        hidden_dim=hidden_dim if hidden_dim else output_dim,
        output_dim=output_dim,
    )
    mlp_dynamic = None
    if ckpt.get('dual_mlp') or ('mlp_stable' in ckpt and 'mlp_dynamic' in ckpt):
        mlp.load_state_dict(ckpt['mlp_stable'])
        mlp_dynamic = MLPProjection(
            input_dim=input_dim,
            hidden_dim=hidden_dim if hidden_dim else output_dim,
            output_dim=output_dim,
        )
        mlp_dynamic.load_state_dict(ckpt['mlp_dynamic'])
        mlp_dynamic = mlp_dynamic.to(device).eval()
    else:
        mlp.load_state_dict(ckpt['mlp'])
    mlp = mlp.to(device).eval()
    print('Stage2 modules loaded.')

    if args.embeddings_path:
        embeddings = load_embedding_tensor(
            args.embeddings_path,
            expected_dim=emb_dim,
            allow_dim_mismatch=args.allow_embedding_dim_mismatch,
        )
    else:
        with open(args.data_path, encoding='utf-8') as source:
            data_for_encode = [json.loads(line) for line in source if line.strip()]
        embeddings = encode_profiles_online(data_for_encode, contriever_model, contriever_tokenizer, device)
        if embeddings.shape[1] != emb_dim:
            raise ValueError(
                f"Online encoder produced dim={embeddings.shape[1]}, checkpoint expects {emb_dim}"
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
                batch, llm_model, qformer, dca, mlp,
                pq_model, contriever_model, contriever_tokenizer,
                tokenizer, num_query_tokens, args.max_new_tokens, device,
                task=task, ablation=ablation, max_length=max_length,
                mlp_dynamic=mlp_dynamic, encoder_type=encoder_type,
            )
            for sid, prompt, ref, pred in zip(
                batch['ids'], batch['prompts'], batch['references'], preds
            ):
                record = {
                    'id': sid,
                    'input': prompt,
                    'output': ref,
                    'reference': ref,
                    'prediction': pred,
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                num_written += 1
            f.flush()
    print(f'Saved {num_written} predictions to {args.output_path}')


def build_parser(default_encoder_type=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',       required=True)
    parser.add_argument('--embeddings_path', default=None)
    parser.add_argument('--llm_path',        required=True)
    parser.add_argument('--codebook_path',   default=None)
    parser.add_argument('--encoder_path',    required=True)
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--output_path',     required=True)
    parser.add_argument('--batch_size',      type=int, default=4)
    parser.add_argument('--max_new_tokens',  type=int, default=128)
    parser.add_argument('--max_length',      type=int, default=None,
                        help='Maximum tokenized inference prompt length; defaults to checkpoint value')
    parser.add_argument('--max_profile_len', type=int, default=None)
    parser.add_argument('--task',            type=str, default=None)
    parser.add_argument('--ablation',        type=str, default=None,
                        choices=['both', 'stable_only', 'dynamic_only'])
    parser.add_argument('--allow_embedding_dim_mismatch', action='store_true')
    parser.add_argument('--allow_checkpoint_config_override', action='store_true')
    parser.add_argument('--encoder_type', choices=['contriever', 'roberta', 'bge'],
                        default=default_encoder_type)
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    main(args)
