"""
Multi-GPU parallel encoder for profile embeddings.
Supports contriever (mean pool), bge (CLS+L2), bge_flag (FlagModel dense_vecs), roberta (mean pool).
Use --shard_id / --num_shards to split work across GPUs; saves to
  {output_pt}.shard{shard_id}of{num_shards}
then run merge_shards.py to produce the final file.
If --num_shards 1 (default), writes directly to --output_pt.
"""
import os
import json
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertTokenizer

from encode import build_sentences_from_jsonl, mean_pooling


def encode_bge(sentences, model_dir, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()

    all_embs = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding BGE", unit="batch"):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            emb = F.normalize(out.last_hidden_state[:, 0], p=2, dim=1)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


def encode_bge_flag(sentences, model_dir, batch_size):
    """Encode using FlagEmbedding FlagModel with max_length=10.
    Handles both dict output (>=1.3, dense_vecs key) and direct ndarray output (1.2.x).
    """
    import numpy as np
    from FlagEmbedding import FlagModel
    model = FlagModel(model_dir, use_fp16=True)

    all_embs = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding BGE (FlagModel)", unit="batch"):
        batch = sentences[i : i + batch_size]
        out = model.encode(batch, max_length=512)
        if isinstance(out, dict):
            emb = out['dense_vecs']
        else:
            emb = out
        all_embs.append(torch.tensor(np.array(emb)))
    return torch.cat(all_embs, dim=0)


def encode_roberta(sentences, model_dir, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()

    all_embs = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding RoBERTa", unit="batch"):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            emb = mean_pooling(out, enc["attention_mask"])
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


def encode_contriever(sentences, model_dir, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()

    all_embs = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding Contriever", unit="batch"):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            emb = mean_pooling(out, enc["attention_mask"])
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


ENCODERS = {
    "contriever": encode_contriever,
    "bge": encode_bge,
    "bge_flag": encode_bge_flag,
    "roberta": encode_roberta,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl",  required=True)
    parser.add_argument("--model_dir",    required=True)
    parser.add_argument("--output_pt",    required=True)
    parser.add_argument("--model_type",   required=True, choices=list(ENCODERS.keys()))
    parser.add_argument("--batch_size",   type=int, default=256)
    parser.add_argument("--shard_id",     type=int, default=0)
    parser.add_argument("--num_shards",   type=int, default=1)
    args = parser.parse_args()

    print(f"Reading JSONL: {args.input_jsonl}")
    orders, sentences = build_sentences_from_jsonl(args.input_jsonl)
    print(f"Total items: {len(orders)}")

    # Take this shard's slice (interleaved for even load balancing)
    if args.num_shards > 1:
        orders    = orders[args.shard_id :: args.num_shards]
        sentences = sentences[args.shard_id :: args.num_shards]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(orders)} items")

    encode_fn = ENCODERS[args.model_type]
    embeddings = encode_fn(sentences, args.model_dir, args.batch_size)
    print(f"Encoded shape: {embeddings.shape}")

    if args.num_shards > 1:
        out_path = f"{args.output_pt}.shard{args.shard_id}of{args.num_shards}"
    else:
        out_path = args.output_pt

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save({"orders": orders, "embeddings": embeddings}, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
