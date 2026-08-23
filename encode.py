import os
import json
import argparse
from typing import List, Tuple

os.environ["TRANSFORMERS_NO_FAST_TOKENIZER"] = "1"

import torch
from tqdm import tqdm
from transformers import AutoModel, BertTokenizer


_ID_KEYS = {"id", "uid", "user_id", "item_id", "order"}

_COMBINATION_TEMPLATES = {
    frozenset(["title", "text"]):
        "The user wrote a title \"{title}\" for the article: {text}",
    frozenset(["title", "abstract"]):
        "The user wrote a paper titled \"{title}\" with the following abstract: {abstract}",
    frozenset(["tag", "description"]):
        "The user reviewed a {tag} with the following description: {description}",
    frozenset(["text", "score"]):
        "The user gave a score of {score} for the following review: {text}",
    frozenset(["text"]):
        "The user wrote: {text}",
    frozenset(["title"]):
        "The user wrote a piece titled \"{title}\".",
    frozenset(["description"]):
        "The user wrote: {description}",
    frozenset(["abstract"]):
        "The user wrote an abstract: {abstract}",
    frozenset(["description", "overall", "reviewText", "summary"]):
        "The user rated {overall}/5 for a product described as: {description}. "
        "They wrote: {reviewText} Summary: {summary}",
    frozenset(["overall", "reviewText", "summary"]):
        "The user gave an overall rating of {overall} and wrote: {reviewText} Summary: {summary}",
    frozenset(["overall", "reviewText"]):
        "The user gave an overall rating of {overall} and wrote: {reviewText}",
    frozenset(["reviewText", "summary"]):
        "The user wrote a review: {reviewText} Summary: {summary}",
    frozenset(["reviewText"]):
        "The user wrote a review: {reviewText}",
    frozenset(["summary"]):
        "The user wrote a summary: {summary}",
}


def connect_fields(fields: List[Tuple[str, str]]) -> str:
    if not fields:
        return ""

    field_dict = {k: v for k, v in fields}
    key_set = frozenset(field_dict.keys())

    template = _COMBINATION_TEMPLATES.get(key_set)
    if template:
        return template.format(**field_dict)

    parts = [f"{k}: {v.strip()}" for k, v in fields]
    return "The user wrote " + ". ".join(parts) + "."


def profile_item_to_text(item: dict) -> str:
    fields = [
        (k, v.strip())
        for k, v in item.items()
        if k.lower() not in _ID_KEYS and isinstance(v, str) and v.strip()
    ]
    return connect_fields(fields)


def build_sentences_from_jsonl(path: str) -> Tuple[List[int], List[str]]:
    items = []

    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading JSONL", unit="line"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            profile = data.get("profile", [])
            if not isinstance(profile, list):
                continue

            for profile_item in profile:
                if not isinstance(profile_item, dict):
                    continue
                order = profile_item.get("order")
                if order is None:
                    continue
                text = profile_item_to_text(profile_item)
                if not text:
                    continue
                items.append((order, text))

    items.sort(key=lambda x: x[0])
    orders = [x[0] for x in items]
    sentences = [x[1] for x in items]
    return orders, sentences


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(1)
    sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def encode_with_contriever(
    sentences: List[str],
    model_dir: str,
    batch_size: int,
) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(
        model_dir,
        do_lower_case=True,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    all_embeddings = []

    for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding", unit="batch"):
        batch_texts = sentences[i:i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            embeddings = mean_pooling(outputs, encoded["attention_mask"])

        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_pt", type=str, required=True)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()

    orders, sentences = build_sentences_from_jsonl(args.input_jsonl)

    embeddings = encode_with_contriever(sentences, args.model_dir, batch_size=args.batch_size)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_pt)), exist_ok=True)
    torch.save(
        {
            "orders": orders,
            "embeddings": embeddings,
        },
        args.output_pt,
    )


if __name__ == "__main__":
    main()