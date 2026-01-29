import os
import json
from typing import List, Tuple

# 禁用 fast tokenizer，强制使用 slow 版本，避免缺少 sentencepiece/tiktoken 报错
os.environ["TRANSFORMERS_NO_FAST_TOKENIZER"] = "1"

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertTokenizer


INPUT_JSONL = "/remote-home/share/lwang_share/data/alignx/alignx_train_unique_labeled.jsonl"
# 实际模型目录在 facebook/contriever 子目录下
MODEL_DIR = "/remote-home/share/lwang_share/wangliang/model/contriever/facebook/contriever"
OUTPUT_PT = "/remote-home/share/lwang_share/data/alignx/alignx_ugc_contriever.pt"


def build_sentences_from_jsonl(path: str) -> Tuple[List[int], List[str]]:
    """
    从带 ugc_id 的 jsonl 中读取数据，按 ugc_id 排序后返回：
    - ugc_ids: List[int]
    - sentences: List[str]，形式为
      \"The user respond {comment} for question {prompt}\"
    """
    ugc_items = []

    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="读取 JSONL", unit="line"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            ugc_list = data.get("User-Generated Content", [])
            if not isinstance(ugc_list, list):
                continue

            for item in ugc_list:
                if not isinstance(item, dict):
                    continue
                ugc_id = item.get("ugc_id")
                comment = item.get("comment")
                prompt = item.get("prompt")
                if ugc_id is None or comment is None or prompt is None:
                    continue

                text = f"The user respond {comment} for question {prompt}"
                ugc_items.append((ugc_id, text))

    # 按 ugc_id 排序，保证编码顺序与 label 顺序一致
    ugc_items.sort(key=lambda x: x[0])
    ugc_ids = [x[0] for x in ugc_items]
    sentences = [x[1] for x in ugc_items]
    return ugc_ids, sentences


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # [B, L, H]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(1)
    sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def encode_with_contriever(
    sentences: List[str],
    model_dir: str,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    使用 contriever 对句子进行编码，返回 [N, H] 的张量。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 有些本地模型没有 fast tokenizer 配置，直接用 BertTokenizer（slow）
    tokenizer = BertTokenizer.from_pretrained(
        model_dir,
        do_lower_case=True,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    all_embeddings = []

    for i in tqdm(range(0, len(sentences), batch_size), desc="编码 UGC", unit="batch"):
        batch_texts = sentences[i : i + batch_size]
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
    print(f"读取数据自: {INPUT_JSONL}")
    ugc_ids, sentences = build_sentences_from_jsonl(INPUT_JSONL)
    print(f"总共找到 {len(sentences)} 条 UGC 需要编码。")

    print(f"加载 contriever 模型自: {MODEL_DIR}")
    embeddings = encode_with_contriever(sentences, MODEL_DIR, batch_size=128)
    print(f"编码完成，得到向量尺寸: {embeddings.shape}")

    os.makedirs(os.path.dirname(OUTPUT_PT), exist_ok=True)
    torch.save(
        {
            "ugc_ids": ugc_ids,
            "embeddings": embeddings,
        },
        OUTPUT_PT,
    )
    print(f"已保存到: {OUTPUT_PT}")


if __name__ == "__main__":
    main()

