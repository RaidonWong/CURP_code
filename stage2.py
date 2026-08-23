import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BertTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
import argparse
from typing import List, Dict, Any, Iterable, Optional, Sequence, Tuple
import numpy as np


STABLE_TOKEN = "<USR_STABLE>"
DYNAMIC_TOKEN = "<USR_DYN>"
CHECKPOINT_FORMAT_VERSION = 3


def create_accelerator(args):
    set_seed(getattr(args, "seed", 42), device_specific=True)
    return Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )


def ensure_chat_template(tokenizer, model_path):
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        print(f"✅ tokenizer已有 chat_template")
        return
    for name in ("chat_template.jinja", "chat_template.txt", "chat_template.json"):
        template_path = os.path.join(model_path, name)
        if os.path.exists(template_path):
            try:
                with open(template_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if name.endswith(".json"):
                    parsed = json.loads(content)
                    content = parsed.get("chat_template", parsed) if isinstance(parsed, dict) else parsed
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("chat template file did not contain a non-empty string")
                tokenizer.chat_template = content
                print(f"✅ 从文件加载 chat_template: {template_path}")
                return
            except Exception as e:
                print(f"⚠️  读取 chat_template 失败: {template_path} -> {e}")
    raise FileNotFoundError(
        f"Tokenizer has no chat_template and none was found under {model_path}"
    )


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(1)
    sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def _as_token_id_list(value) -> List[int]:
    """Normalize tokenizer/chat-template outputs to a flat list of token ids."""
    if isinstance(value, dict):
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError(f"Expected one tokenized sample, got batch size {len(value)}")
        value = value[0]
    return [int(token_id) for token_id in value]


def expected_placeholder_count(ablation: str, num_query_tokens: int) -> int:
    if ablation == "stable_only":
        return num_query_tokens
    if ablation == "dynamic_only":
        return 1
    if ablation == "both":
        return num_query_tokens + 1
    raise ValueError(f"Unknown ablation: {ablation}")


def collect_placeholder_positions(
    input_ids: Sequence[int] | torch.Tensor,
    stable_id: int,
    dynamic_id: int,
    ablation: str,
    num_query_tokens: int,
    sample_label: str = "sample",
) -> List[int]:
    ids = _as_token_id_list(input_ids)
    stable_pos = [i for i, token_id in enumerate(ids) if token_id == stable_id]
    dynamic_pos = [i for i, token_id in enumerate(ids) if token_id == dynamic_id]
    if ablation == "stable_only":
        positions = stable_pos
        unexpected = dynamic_pos
    elif ablation == "dynamic_only":
        positions = dynamic_pos
        unexpected = stable_pos
    else:
        positions = sorted(stable_pos + dynamic_pos)
        unexpected = []

    expected = expected_placeholder_count(ablation, num_query_tokens)
    if len(positions) != expected or unexpected:
        raise ValueError(
            f"{sample_label}: placeholder mismatch for ablation={ablation}; "
            f"expected {expected}, found stable={len(stable_pos)}, dynamic={len(dynamic_pos)}. "
            "The chat template or truncation removed/duplicated a user token."
        )
    return positions


def _truncate_prompt_preserving_placeholders(
    prompt_ids: List[int],
    max_prompt_tokens: int,
    placeholder_positions: Sequence[int],
) -> List[int]:
    """Middle-truncate a chat prompt while retaining user tokens and assistant header."""
    if len(prompt_ids) <= max_prompt_tokens:
        return prompt_ids
    if max_prompt_tokens <= 0:
        raise ValueError("No token budget remains for the prompt")
    if not placeholder_positions:
        return prompt_ids[-max_prompt_tokens:]

    prefix_len = max(placeholder_positions) + 1
    if prefix_len >= max_prompt_tokens:
        raise ValueError(
            f"max_length leaves only {max_prompt_tokens} prompt tokens, but the required "
            f"placeholder prefix needs at least {prefix_len + 1}. Increase --max_length."
        )
    tail_len = max_prompt_tokens - prefix_len
    return prompt_ids[:prefix_len] + prompt_ids[-tail_len:]


def tokenize_generation_prompt(
    tokenizer,
    user_prompt: str,
    max_length: int,
    stable_id: int,
    dynamic_id: int,
    ablation: str,
    num_query_tokens: int,
    sample_label: str = "sample",
) -> Tuple[torch.Tensor, List[int]]:
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    prompt_ids = _as_token_id_list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    positions = collect_placeholder_positions(
        prompt_ids, stable_id, dynamic_id, ablation, num_query_tokens, sample_label
    )
    prompt_ids = _truncate_prompt_preserving_placeholders(
        prompt_ids, max_length, positions
    )
    positions = collect_placeholder_positions(
        prompt_ids, stable_id, dynamic_id, ablation, num_query_tokens, sample_label
    )
    return torch.tensor(prompt_ids, dtype=torch.long), positions


def tokenize_training_example(
    tokenizer,
    user_prompt: str,
    answer: str,
    max_length: int,
    stable_id: int,
    dynamic_id: int,
    ablation: str,
    num_query_tokens: int,
    sample_label: str = "sample",
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Build exact assistant-only supervision without model-specific boundary guesses."""
    if max_length < 2:
        raise ValueError(f"max_length must be at least 2, got {max_length}")

    prompt_ids = _as_token_id_list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    positions = collect_placeholder_positions(
        prompt_ids, stable_id, dynamic_id, ablation, num_query_tokens, sample_label
    )

    answer_ids = _as_token_id_list(
        tokenizer(answer, add_special_tokens=False)["input_ids"]
    )
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and (not answer_ids or answer_ids[-1] != eos_id):
        answer_ids.append(int(eos_id))
    if not answer_ids:
        raise ValueError(
            f"{sample_label}: answer tokenization is empty and tokenizer has no EOS token"
        )

    # Keep enough prompt prefix for all placeholders plus at least one suffix token.
    placeholder_prefix = max(positions) + 1
    # Retain enough of the chat-template suffix to include the assistant role
    # header, rather than keeping only its final delimiter token.
    min_tail_tokens = min(16, len(prompt_ids) - placeholder_prefix)
    min_prompt_tokens = placeholder_prefix + max(1, min_tail_tokens)
    if min_prompt_tokens >= max_length:
        raise ValueError(
            f"{sample_label}: max_length={max_length} is too small for the chat prompt "
            f"and {len(positions)} placeholder tokens"
        )
    max_answer_tokens = max_length - min_prompt_tokens
    if len(answer_ids) > max_answer_tokens:
        answer_ids = answer_ids[:max_answer_tokens]
        if eos_id is not None:
            answer_ids[-1] = int(eos_id)
    prompt_budget = max_length - len(answer_ids)
    prompt_ids = _truncate_prompt_preserving_placeholders(
        prompt_ids, prompt_budget, positions
    )
    positions = collect_placeholder_positions(
        prompt_ids, stable_id, dynamic_id, ablation, num_query_tokens, sample_label
    )

    input_ids = torch.tensor(prompt_ids + answer_ids, dtype=torch.long)
    labels = torch.tensor([-100] * len(prompt_ids) + answer_ids, dtype=torch.long)
    if not torch.any(labels != -100):
        raise ValueError(f"{sample_label}: example has no supervised assistant tokens")
    return input_ids, labels, positions


def left_pad_tokenized_batch(
    input_ids_list: Sequence[torch.Tensor],
    labels_list: Optional[Sequence[torch.Tensor]],
    placeholder_positions: Sequence[Sequence[int]],
    pad_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, List[List[int]]]:
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id")
    max_len = max(len(ids) for ids in input_ids_list)
    padded_ids, padded_labels, attention_masks, adjusted_positions = [], [], [], []
    for index, ids in enumerate(input_ids_list):
        pad_len = max_len - len(ids)
        padded_ids.append(F.pad(ids, (pad_len, 0), value=pad_token_id))
        attention_masks.append(
            F.pad(torch.ones(len(ids), dtype=torch.long), (pad_len, 0), value=0)
        )
        adjusted_positions.append([position + pad_len for position in placeholder_positions[index]])
        if labels_list is not None:
            padded_labels.append(F.pad(labels_list[index], (pad_len, 0), value=-100))

    input_ids = torch.stack(padded_ids).to(device)
    attention_mask = torch.stack(attention_masks).to(device)
    labels = torch.stack(padded_labels).to(device) if labels_list is not None else None
    return input_ids, labels, attention_mask, adjusted_positions


def inject_placeholder_embeddings(
    llm_model,
    input_ids: torch.Tensor,
    user_embeddings: torch.Tensor,
    placeholder_positions: Sequence[Sequence[int]],
) -> torch.Tensor:
    batch_size, seq_len = input_ids.shape
    expected_tokens = user_embeddings.shape[1]
    for sample_index, positions in enumerate(placeholder_positions):
        if len(positions) != expected_tokens:
            raise ValueError(
                f"batch item {sample_index}: expected {expected_tokens} placeholder positions, "
                f"found {len(positions)}"
            )

    base_embeddings = llm_model.get_input_embeddings()(input_ids).detach()
    user_embeddings = user_embeddings.to(
        device=base_embeddings.device, dtype=base_embeddings.dtype
    )
    batch_indices = torch.arange(batch_size, device=input_ids.device).repeat_interleave(
        expected_tokens
    )
    sequence_indices = torch.tensor(
        [position for positions in placeholder_positions for position in positions],
        device=input_ids.device,
        dtype=torch.long,
    )
    token_indices = torch.arange(expected_tokens, device=input_ids.device).repeat(batch_size)
    output = base_embeddings.clone()
    output[batch_indices, sequence_indices] = user_embeddings[batch_indices, token_indices]
    return output


def load_embedding_tensor(
    embeddings_path: str,
    expected_dim: Optional[int] = None,
    allow_dim_mismatch: bool = False,
) -> torch.Tensor:
    raw = torch.load(embeddings_path, map_location="cpu")
    if isinstance(raw, torch.Tensor):
        embeddings = raw
    elif isinstance(raw, list):
        embeddings = torch.stack(raw)
    elif isinstance(raw, dict) and "embeddings" in raw:
        embeddings = raw["embeddings"]
    else:
        raise ValueError(f"Unknown embeddings format in {embeddings_path}: {type(raw)}")
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding tensor in {embeddings_path}, got {tuple(embeddings.shape)}"
        )
    if not torch.is_floating_point(embeddings):
        raise ValueError(f"Embedding tensor must use a floating dtype, got {embeddings.dtype}")
    embeddings = embeddings.float().contiguous()
    if expected_dim is not None and embeddings.shape[1] != expected_dim:
        if not allow_dim_mismatch:
            raise ValueError(
                f"Embedding dimension mismatch: file={embeddings.shape[1]}, expected={expected_dim}. "
                "Use embeddings/codebook produced by the same encoder, or explicitly pass "
                "--allow_embedding_dim_mismatch for legacy compatibility."
            )
        print(
            f"WARNING: adapting embedding dimension from {embeddings.shape[1]} to {expected_dim}; "
            "this compatibility mode can change model semantics."
        )
        if embeddings.shape[1] > expected_dim:
            embeddings = embeddings[:, :expected_dim]
        else:
            embeddings = F.pad(embeddings, (0, expected_dim - embeddings.shape[1]))
    return embeddings


def validate_encoder_dimension(model, expected_dim: int, label: str = "encoder"):
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is not None and int(hidden_size) != int(expected_dim):
        raise ValueError(
            f"{label} hidden_size={hidden_size} does not match profile/codebook "
            f"dimension={expected_dim}"
        )


class PQCodebookModel(nn.Module):
    """Quantizer for stage-2. Supports three quant_type values (rebuttal Claim 2):
      - "pq"  (default): product quantization over `num_subspaces` subspaces.
      - "rq" : residual quantization — full-dim codebooks applied in stages.
      - "opq": PQ on a rotated space (R baked in the checkpoint), rotated back.
    Downstream code only consumes the reconstructed full-dim `quantized` tensor,
    so the rest of stage-2/inference is unchanged across quant types."""
    def __init__(self, codebook_path, device="cpu"):
        super().__init__()
        checkpoint = torch.load(codebook_path, map_location=device)
        if "codebooks" not in checkpoint:
            raise ValueError(f"Checkpoint must contain 'codebooks' key. Found: {checkpoint.keys()}")

        self.quant_type = checkpoint.get("quant_type", "pq")
        if self.quant_type not in {"pq", "rq", "opq"}:
            raise ValueError(f"Unsupported quant_type={self.quant_type!r}")

        codebooks_list = []
        for cb in checkpoint["codebooks"]:
            param = nn.Parameter(
                cb.to(device=device, dtype=torch.float32)
                if isinstance(cb, torch.Tensor)
                else torch.tensor(cb, device=device, dtype=torch.float32),
                requires_grad=False
            )
            codebooks_list.append(param)
        self.codebooks = nn.ParameterList(codebooks_list)
        if not self.codebooks:
            raise ValueError("Codebook checkpoint contains an empty codebooks list")
        if any(codebook.ndim != 2 for codebook in self.codebooks):
            raise ValueError("Every codebook must be a 2D tensor [entries, dimension]")

        if self.quant_type == "opq":
            if "rotation" not in checkpoint:
                raise ValueError("OPQ checkpoint is missing the rotation matrix")
            R = checkpoint["rotation"]
            R = (
                R.to(device=device, dtype=torch.float32)
                if isinstance(R, torch.Tensor)
                else torch.tensor(R, device=device, dtype=torch.float32)
            )
            self.rotation = nn.Parameter(R, requires_grad=False)          # (D, D)
            self.num_subspaces = checkpoint.get("num_subspaces", len(self.codebooks))
            self.subspace_dim  = checkpoint.get("subspace_dim", self.codebooks[0].shape[1])
            self.codebook_size = self.codebooks[0].shape[0]
            self.emb_dim = self.num_subspaces * self.subspace_dim
            if tuple(self.rotation.shape) != (self.emb_dim, self.emb_dim):
                raise ValueError(
                    f"OPQ rotation shape {tuple(self.rotation.shape)} does not match "
                    f"embedding dimension {self.emb_dim}"
                )
            print(f"Loaded OPQ codebook: {self.num_subspaces} subspaces × {self.subspace_dim}D, "
                  f"{self.codebook_size} entries + rotation {tuple(self.rotation.shape)}")
        elif self.quant_type == "rq":
            self.num_stages    = len(self.codebooks)
            self.codebook_size = self.codebooks[0].shape[0]
            self.emb_dim       = self.codebooks[0].shape[1]
            if any(codebook.shape[1] != self.emb_dim for codebook in self.codebooks):
                raise ValueError("All RQ stages must use the same full embedding dimension")
            print(f"Loaded RQ codebook: {self.num_stages} stages × {self.codebook_size} "
                  f"entries × {self.emb_dim}D")
        else:  # pq
            self.num_subspaces = checkpoint.get("num_subspaces", len(self.codebooks))
            self.subspace_dim  = checkpoint.get("subspace_dim", self.codebooks[0].shape[1])
            self.codebook_size = self.codebooks[0].shape[0]
            self.emb_dim = self.num_subspaces * self.subspace_dim
            if len(self.codebooks) != self.num_subspaces:
                raise ValueError(
                    f"PQ metadata says {self.num_subspaces} subspaces but checkpoint has "
                    f"{len(self.codebooks)} codebooks"
                )
            if any(codebook.shape[1] != self.subspace_dim for codebook in self.codebooks):
                raise ValueError("PQ codebook subspace dimensions are inconsistent")
            print(f"Loaded PQ codebook: {self.num_subspaces} subspaces × {self.subspace_dim}D, {self.codebook_size} entries")

    @staticmethod
    def _nearest_indices(vectors, codebook):
        # Squared Euclidean distance avoids the sqrt and large temporary used by cdist.
        vectors_f = vectors.float()
        codebook_f = codebook.float()
        distances = (
            vectors_f.square().sum(dim=-1, keepdim=True)
            + codebook_f.square().sum(dim=-1).unsqueeze(0)
            - 2.0 * vectors_f @ codebook_f.t()
        )
        return distances.argmin(dim=-1)

    def _pq_quantize(self, flat_embs, num_subspaces, subspace_dim):
        subspace_embs = flat_embs.reshape(-1, num_subspaces, subspace_dim)
        quantized_parts, all_indices = [], []
        for i, codebook in enumerate(self.codebooks):
            subspace = subspace_embs[:, i, :]
            codebook = codebook.to(subspace.device)
            indices = self._nearest_indices(subspace, codebook)
            quantized_parts.append(codebook[indices])
            all_indices.append(indices)
        quantized = torch.cat(quantized_parts, dim=-1)
        all_indices = torch.stack(all_indices, dim=-1)
        return quantized, all_indices

    def quantize(self, embeddings):
        if embeddings.ndim != 3:
            raise ValueError(
                f"Expected profile embeddings [batch, profile, dim], got {tuple(embeddings.shape)}"
            )
        batch_size, seq_len, emb_dim = embeddings.shape
        if emb_dim != self.emb_dim:
            raise ValueError(
                f"Quantizer expects embedding dimension {self.emb_dim}, got {emb_dim}"
            )
        flat_embs = embeddings.reshape(-1, emb_dim)

        if self.quant_type == "opq":
            R = self.rotation.to(flat_embs.device)
            rotated = flat_embs @ R                                   # (N, D)
            q_rot, all_indices = self._pq_quantize(rotated, self.num_subspaces, self.subspace_dim)
            quantized = q_rot @ R.t()                                 # rotate back to original space
            quantized = quantized.view(batch_size, seq_len, emb_dim)
            all_indices = all_indices.view(batch_size, seq_len, self.num_subspaces)
            return quantized, all_indices

        if self.quant_type == "rq":
            residual = flat_embs
            q = torch.zeros_like(flat_embs)
            all_indices = []
            for codebook in self.codebooks:
                codebook = codebook.to(residual.device)
                indices = self._nearest_indices(residual, codebook)
                sel = codebook[indices]
                q = q + sel
                residual = residual - sel
                all_indices.append(indices)
            quantized = q.view(batch_size, seq_len, emb_dim)
            all_indices = torch.stack(all_indices, dim=-1).view(batch_size, seq_len, self.num_stages)
            return quantized, all_indices

        # pq (default)
        quantized, all_indices = self._pq_quantize(flat_embs, self.num_subspaces, self.subspace_dim)
        quantized = quantized.view(batch_size, seq_len, emb_dim)
        all_indices = all_indices.view(batch_size, seq_len, self.num_subspaces)
        return quantized, all_indices

    def forward(self, embeddings):
        return self.quantize(embeddings)


class IdentityCodebookModel(nn.Module):
    """No-PQ ablation with the same interface as the quantized path."""

    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.quant_type = "none"

    def forward(self, embeddings):
        if embeddings.shape[-1] != self.emb_dim:
            raise ValueError(
                f"No-PQ path expects embedding dimension {self.emb_dim}, got {embeddings.shape[-1]}"
            )
        return embeddings, None


class QFormerLayer(nn.Module):
    def __init__(self, emb_dim=768, num_heads=8):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, queries, context, context_key_padding_mask=None):
        q2, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + q2)
        q2, _ = self.cross_attn(
            queries, context, context,
            key_padding_mask=context_key_padding_mask
        )
        queries = self.norm2(queries + q2)
        return queries


class QFormer(nn.Module):
    def __init__(self, num_query_tokens=4, emb_dim=768, num_heads=8, num_layers=2):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, emb_dim) * 0.02)
        self.layers = nn.ModuleList([
            QFormerLayer(emb_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, profile_embeddings, key_padding_mask=None):
        batch_size = profile_embeddings.size(0)
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        for layer in self.layers:
            queries = layer(queries, profile_embeddings, context_key_padding_mask=key_padding_mask)
        return queries


class MeanPoolStable(nn.Module):
    """Parameter-free stable branch used by the mean-pooling ablation."""

    def __init__(self, emb_dim):
        super().__init__()
        self.register_buffer("query_tokens", torch.empty(1, emb_dim), persistent=False)

    def forward(self, profile_embeddings, key_padding_mask=None):
        if key_padding_mask is None:
            return profile_embeddings.mean(dim=1, keepdim=True)
        valid = (~key_padding_mask).to(profile_embeddings.dtype).unsqueeze(-1)
        pooled = (profile_embeddings * valid).sum(dim=1, keepdim=True)
        denominator = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        return pooled / denominator


class DynamicCrossAttention(nn.Module):
    def __init__(self, emb_dim=768, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, query_emb, profile_embeddings, key_padding_mask=None):
        out, _ = self.cross_attn(
            query_emb, profile_embeddings, profile_embeddings,
            key_padding_mask=key_padding_mask
        )
        return self.norm(query_emb + out)


class MLPProjection(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=None, output_dim=3584):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        print(f"Initialized MLP: {input_dim} -> {hidden_dim} -> {output_dim}")

    def forward(self, x):
        return self.mlp(x)


class Stage2Dataset(Dataset):
    def __init__(self, jsonl_path, embeddings_path, tokenizer,
                 max_profile_len=None, max_length=2048, emb_dim=None,
                 allow_embedding_dim_mismatch=False):
        self.tokenizer = tokenizer
        self.max_profile_len = max_profile_len
        self.max_length = max_length
        self.emb_dim = emb_dim

        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

        print(f"Loading pre-computed embeddings from {embeddings_path}...")
        self.embeddings = load_embedding_tensor(
            embeddings_path,
            expected_dim=self.emb_dim,
            allow_dim_mismatch=allow_embedding_dim_mismatch,
        )
        self.emb_dim = self.embeddings.shape[1]

        print(f"Loaded {len(self.data)} samples, {len(self.embeddings)} embeddings {self.embeddings.shape}")

        for tok in (STABLE_TOKEN, DYNAMIC_TOKEN):
            if tok not in tokenizer.get_vocab():
                raise ValueError(
                    f"Token '{tok}' not in tokenizer vocab. "
                    "Add it with tokenizer.add_tokens([...]) before creating the dataset."
                )

        invalid_orders = 0
        empty_profiles = 0
        for item in self.data:
            profile = item.get("profile", [])
            if self.max_profile_len is not None:
                profile = profile[:self.max_profile_len]
            valid = 0
            for profile_item in profile:
                order = profile_item.get("order")
                if isinstance(order, int) and 0 <= order < len(self.embeddings):
                    valid += 1
                else:
                    invalid_orders += 1
            if valid == 0:
                empty_profiles += 1
        if invalid_orders:
            print(
                f"WARNING: skipped {invalid_orders} profile item(s) with invalid embedding order"
            )
        if empty_profiles:
            print(
                f"WARNING: {empty_profiles} sample(s) have no valid profile embeddings; "
                "a single zero fallback token will be used to keep attention numerically safe"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        profile = item.get("profile", [])
        if self.max_profile_len is not None:
            profile = profile[:self.max_profile_len]

        embs = []
        for p_item in profile:
            order = p_item.get("order")
            if isinstance(order, int) and 0 <= order < len(self.embeddings):
                embs.append(self.embeddings[order])

        if not embs:
            embs.append(torch.zeros(self.emb_dim, dtype=self.embeddings.dtype))

        return {
            "prompt": item.get("input", ""),
            "chosen": item.get("output", ""),
            "profile_embeddings": torch.stack(embs),
        }


def collate_fn(batch):
    prompts  = [b["prompt"] for b in batch]
    chosens  = [b["chosen"] for b in batch]
    embs_list = [b["profile_embeddings"] for b in batch]

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
        "prompt":             prompts,
        "chosen":             chosens,
        "profile_embeddings": padded,
        "profile_mask":       mask,
    }


def _build_user_prompt(task, stable_str, dynamic_token, prompt_text, ablation='both'):
    if task == 'lamp1_min':
        if ablation == 'stable_only':
            return f"{stable_str}\n{prompt_text}"
        elif ablation == 'dynamic_only':
            return f"{dynamic_token}\n{prompt_text}"
        return f"{stable_str} {dynamic_token}\n{prompt_text}"
    elif task == 'lamp3_min':
        stable_line  = f"A user's rating preferences learned from their past reviews are encoded in: {stable_str}"
        dynamic_line = f"The most relevant past reviews are encoded in: {dynamic_token}"
        if ablation == 'stable_only':
            return f"{stable_line}\n{prompt_text}"
        elif ablation == 'dynamic_only':
            return f"{dynamic_line}\n{prompt_text}"
        return f"{stable_line}\n{dynamic_line}\n{prompt_text}"
    elif task == 'lamp1':
        stable_line  = f"The research expertise and dominant scientific topics that define this scholar's work, distilled from their complete publication history, are encoded in: {stable_str}"
        dynamic_line = f"The research aspects most topically aligned with the target paper are encoded in: {dynamic_token}"
        suffix       = "Select the reference whose subject matter best matches this scholar's research domain. Output only [1] or [2]."
    elif task == 'lamp3':
        stable_line  = f"This reviewer's personal rating calibration—how they translate expressed satisfaction into numerical scores—is encoded in: {stable_str}"
        dynamic_line = f"The reviewer's past reviews whose sentiment most closely matches the current review are encoded in: {dynamic_token}"
        suffix       = "Based on this user's personal rating behavior, predict the score they assigned. Output only 1, 2, 3, 4, or 5."
    elif task == 'lamp4':
        stable_line  = f"This author's headline writing style and preferences, learned from their past articles, are encoded in: {stable_str}"
        dynamic_line = f"The aspects of their past articles most relevant to the current article are encoded in: {dynamic_token}"
        suffix       = "Generate a headline in this author's personal writing style. Output only the headline."
    elif task == 'longlamp':
        stable_line  = f"This reviewer's writing style, tone, and vocabulary, learned from their past reviews, are encoded in: {stable_str}"
        dynamic_line = f"The aspects of their reviewing history most relevant to this product are encoded in: {dynamic_token}"
        suffix       = "Write the review in this reviewer's personal writing style. Output only the review text."
    elif task == 'lamp2':
        stable_line  = f"A user's preferences learned from their past writings are encoded in: {stable_str}"
        dynamic_line = f"The aspects of their history most relevant to the current task are encoded in: {dynamic_token}"
        suffix       = "Output only the answer."
    else:
        stable_line  = f"A user's writing style, tone, and preferences learned from their past writings are encoded in: {stable_str}"
        dynamic_line = f"The aspects of their history most relevant to the current task are encoded in: {dynamic_token}"
        suffix       = "Complete the task above in this user's personal writing style. Output only the answer."
    if ablation == 'stable_only':
        return f"{stable_line}\n{prompt_text}\n{suffix}"
    elif ablation == 'dynamic_only':
        return f"{dynamic_line}\n{prompt_text}\n{suffix}"
    return f"{stable_line}\n{dynamic_line}\n{prompt_text}\n{suffix}"


class Stage2Trainer:
    def __init__(self, llm_model, mlp_model, qformer, dynamic_cross_attn,
                 pq_codebook_model, contriever_model, contriever_tokenizer,
                 tokenizer, accelerator, num_query_tokens=4, max_length=2048, task=None,
                 ablation='both', finetune_codebook=False):
        self.llm_model            = llm_model
        self.mlp_model            = mlp_model
        self.qformer              = qformer
        self.dynamic_cross_attn   = dynamic_cross_attn
        self.pq_codebook_model    = pq_codebook_model
        self.contriever_model     = contriever_model
        self.contriever_tokenizer = contriever_tokenizer
        self.tokenizer            = tokenizer
        self.accelerator          = accelerator
        self.num_query_tokens     = num_query_tokens
        self.max_length           = max_length
        self.task                 = task
        self.ablation             = ablation
        self.finetune_codebook    = finetune_codebook
        self.stable_token         = STABLE_TOKEN
        self.dynamic_token        = DYNAMIC_TOKEN

        for param in self.llm_model.parameters():
            param.requires_grad = False
        for param in self.pq_codebook_model.parameters():
            param.requires_grad = finetune_codebook  # rebuttal: fine-tune codebook if requested
        for param in self.contriever_model.parameters():
            param.requires_grad = False
        self.llm_model.eval()
        self.contriever_model.eval()

    def _encode_queries(self, queries, device):
        encoded = self.contriever_tokenizer(
            queries, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs   = self.contriever_model(**encoded)
            query_emb = mean_pooling(outputs, encoded["attention_mask"])
        return query_emb.unsqueeze(1)

    def _project_tokens(self, stable_tokens, dynamic_token):
        if self.ablation == "stable_only":
            return self.mlp_model(stable_tokens)
        if self.ablation == "dynamic_only":
            return self.mlp_model(dynamic_token)
        return self.mlp_model(torch.cat([stable_tokens, dynamic_token], dim=1))

    def compute_loss(self, batch):
        prompts            = batch["prompt"]
        chosens            = batch["chosen"]
        profile_embeddings = batch["profile_embeddings"]
        profile_mask       = batch["profile_mask"]

        device     = profile_embeddings.device
        batch_size = len(prompts)

        _pq_ctx = torch.enable_grad() if self.finetune_codebook else torch.no_grad()
        with _pq_ctx:
            quantized_embs, _ = self.pq_codebook_model(profile_embeddings)

        if self.ablation != 'dynamic_only':
            stable_tokens = self.qformer(
                quantized_embs,
                key_padding_mask=profile_mask.to(device)
            )

        if self.ablation != 'stable_only':
            query_embs    = self._encode_queries(prompts, device)
            if query_embs.shape[-1] != quantized_embs.shape[-1]:
                raise ValueError(
                    f"Dynamic query encoder produced dim={query_embs.shape[-1]}, but profile/codebook "
                    f"dim={quantized_embs.shape[-1]}. Use matching encoder embeddings and codebook."
                )
            query_embs    = query_embs.to(dtype=quantized_embs.dtype)
            dynamic_token = self.dynamic_cross_attn(
                query_embs, quantized_embs,
                key_padding_mask=profile_mask.to(device)
            )

        llm_embs = self._project_tokens(
            stable_tokens if self.ablation != "dynamic_only" else None,
            dynamic_token if self.ablation != "stable_only" else None,
        )

        stable_id  = self.tokenizer.convert_tokens_to_ids(self.stable_token)
        dynamic_id = self.tokenizer.convert_tokens_to_ids(self.dynamic_token)

        input_ids_list, labels_list, ph_positions_list = [], [], []

        for i in range(batch_size):
            stable_str = " ".join([self.stable_token] * self.num_query_tokens)
            user_prompt_text = _build_user_prompt(
                self.task, stable_str, self.dynamic_token, prompts[i], ablation=self.ablation
            )
            input_ids, labels, ph_positions = tokenize_training_example(
                self.tokenizer,
                user_prompt_text,
                chosens[i],
                self.max_length,
                stable_id,
                dynamic_id,
                self.ablation,
                self.num_query_tokens,
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

        outputs = self.llm_model(
            inputs_embeds=input_embs,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss


def train(args):
    accelerator = create_accelerator(args)

    if accelerator.is_local_main_process:
        print(f"Stage 2: Training Q-Former (stable) + DynamicCrossAttn (dynamic) + MLP")
        print(f"  Stable tokens (m): {args.num_query_tokens}")
        print(f"  Q-Former layers:   {args.qformer_layers}")
        print(f"  Dynamic tokens:    1")

    print(f"Loading LLM from {args.llm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    ensure_chat_template(tokenizer, args.llm_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    new_tokens = ["<USR_STABLE>", "<USR_DYN>"]
    added = tokenizer.add_tokens(new_tokens)
    print(f"Added {added} placeholder token(s): {new_tokens}")

    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    llm_model.resize_token_embeddings(len(tokenizer))
    llm_model.eval()
    print(f"LLM embedding size: {len(tokenizer)}")

    llm_dim = llm_model.config.hidden_size
    print(f"LLM hidden size: {llm_dim}")

    print(f"Loading Contriever from {args.encoder_path}...")
    os.environ["TRANSFORMERS_NO_FAST_TOKENIZER"] = "1"
    contriever_tokenizer = AutoTokenizer.from_pretrained(
        args.encoder_path, local_files_only=True
    )
    contriever_model = AutoModel.from_pretrained(args.encoder_path)
    contriever_model.eval()
    for param in contriever_model.parameters():
        param.requires_grad = False
    print("Contriever loaded and frozen.")

    print(f"Loading PQ codebook from {args.codebook_path}...")
    pq_codebook_model = PQCodebookModel(args.codebook_path, device="cpu")
    for param in pq_codebook_model.parameters():
        param.requires_grad = False

    emb_dim = pq_codebook_model.emb_dim
    print(f"PQ codebook emb_dim: {emb_dim}")
    validate_encoder_dimension(contriever_model, emb_dim, "Contriever")

    qformer = QFormer(
        num_query_tokens=args.num_query_tokens,
        emb_dim=emb_dim,
        num_heads=args.num_heads,
        num_layers=args.qformer_layers
    )
    print(f"QFormer: {args.num_query_tokens} query tokens, {args.qformer_layers} layers, {args.num_heads} heads")

    dynamic_cross_attn = DynamicCrossAttention(emb_dim=emb_dim, num_heads=args.num_heads)
    print(f"DynamicCrossAttention: emb_dim={emb_dim}, num_heads={args.num_heads}")

    hidden_dim = args.hidden_dim if args.hidden_dim else llm_dim
    mlp_model  = MLPProjection(input_dim=emb_dim, hidden_dim=hidden_dim, output_dim=llm_dim)

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

    trainer = Stage2Trainer(
        llm_model=llm_model,
        mlp_model=mlp_model,
        qformer=qformer,
        dynamic_cross_attn=dynamic_cross_attn,
        pq_codebook_model=pq_codebook_model,
        contriever_model=contriever_model,
        contriever_tokenizer=contriever_tokenizer,
        tokenizer=tokenizer,
        accelerator=accelerator,
        num_query_tokens=args.num_query_tokens,
        max_length=args.max_length,
        task=args.task,
        ablation=args.ablation,
        finetune_codebook=args.finetune_codebook,
    )

    trained_params = (
        list(qformer.parameters()) +
        list(dynamic_cross_attn.parameters()) +
        list(mlp_model.parameters())
    )
    if args.finetune_codebook:
        for p in pq_codebook_model.parameters():
            p.requires_grad = True
        trained_params = trained_params + list(pq_codebook_model.parameters())
        print(f"[finetune_codebook] added {sum(p.numel() for p in pq_codebook_model.parameters())} codebook params to optimizer")
    optimizer = torch.optim.AdamW(
        trained_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    if args.finetune_codebook:
        (
            qformer,
            dynamic_cross_attn,
            mlp_model,
            pq_codebook_model,
            optimizer,
            dataloader,
        ) = accelerator.prepare(
            qformer,
            dynamic_cross_attn,
            mlp_model,
            pq_codebook_model,
            optimizer,
            dataloader,
        )
    else:
        qformer, dynamic_cross_attn, mlp_model, optimizer, dataloader = accelerator.prepare(
            qformer, dynamic_cross_attn, mlp_model, optimizer, dataloader
        )
    trainer.qformer            = qformer
    trainer.dynamic_cross_attn = dynamic_cross_attn
    trainer.mlp_model          = mlp_model

    if not args.finetune_codebook:
        pq_codebook_model = pq_codebook_model.to(accelerator.device)
    contriever_model   = contriever_model.to(accelerator.device)
    trainer.pq_codebook_model = pq_codebook_model
    trainer.contriever_model  = contriever_model

    if not hasattr(llm_model, 'hf_device_map'):
        llm_model = llm_model.to(accelerator.device)
    trainer.llm_model = llm_model

    accumulation_models = [qformer, dynamic_cross_attn, mlp_model]
    if args.finetune_codebook:
        accumulation_models.append(pq_codebook_model)
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
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "update": global_step,
                })

            if (
                accelerator.sync_gradients
                and args.save_steps > 0
                and global_step % args.save_steps == 0
            ):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    _save_checkpoint(
                        accelerator, qformer, dynamic_cross_attn, mlp_model,
                        args, global_step, pq_codebook_model=pq_codebook_model,
                        optimizer=optimizer,
                    )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            _save_checkpoint(accelerator, qformer, dynamic_cross_attn, mlp_model,
                             args, step=global_step, epoch=epoch + 1,
                             pq_codebook_model=pq_codebook_model,
                             optimizer=optimizer)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_checkpoint(accelerator, qformer, dynamic_cross_attn, mlp_model,
                         args, step=None, final=True,
                         pq_codebook_model=pq_codebook_model,
                         optimizer=optimizer)


def _safe_unwrap(model):
    # NB: intentionally avoid accelerator.unwrap_model here. In this environment
    # it imports deepspeed, whose CUDA op builder raises MissingCUDAException
    # when CUDA_HOME is unset, crashing the save at the very end of training.
    # We never use deepspeed; strip any DDP/.module wrapping directly instead.
    while hasattr(model, "module"):
        model = model.module
    return model


def _state_dict_to_cpu(model):
    return {
        key: value.detach().cpu()
        for key, value in _safe_unwrap(model).state_dict().items()
    }


def _checkpoint_metadata(args, step, epoch):
    def resource_name(value):
        """Keep checkpoint metadata portable without storing machine paths."""
        if value is None:
            return None
        normalized = os.path.normpath(str(value))
        return os.path.basename(normalized)

    metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": step,
        "epoch": epoch,
        "task": getattr(args, "task", None),
        "ablation": getattr(args, "ablation", "both"),
        "max_length": getattr(args, "max_length", None),
        "max_profile_len": getattr(args, "max_profile_len", None),
        "placeholder_tokens": {
            "stable": STABLE_TOKEN,
            "dynamic": DYNAMIC_TOKEN,
        },
        "llm_name": resource_name(getattr(args, "llm_path", None)),
        "encoder_name": resource_name(getattr(args, "encoder_path", None)),
        "codebook_name": resource_name(getattr(args, "codebook_path", None)),
        "no_pq": bool(getattr(args, "no_pq", False)),
        "stable_pooling": getattr(args, "stable_pooling", "qformer"),
        "gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps", 1),
        "seed": getattr(args, "seed", 42),
    }
    numpy_state = np.random.get_state()
    metadata["rng_state"] = {
        "torch": torch.get_rng_state(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
    }
    if torch.cuda.is_available():
        metadata["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()
    return metadata


def _save_checkpoint(accelerator, qformer, dynamic_cross_attn, mlp_model,
                     args, step, final=False, epoch=None, pq_codebook_model=None,
                     optimizer=None):
    if final:
        tag = "stage2_final"
    elif epoch is not None:
        tag = f"stage2_epoch-{epoch}"
    else:
        tag = f"stage2_checkpoint-{step}"
    path  = os.path.join(args.output_dir, tag)
    os.makedirs(path, exist_ok=True)

    unwrapped_qformer = _safe_unwrap(qformer)
    unwrapped_mlp = _safe_unwrap(mlp_model)

    ckpt = {
        **_checkpoint_metadata(args, step, epoch),
        "qformer":           _state_dict_to_cpu(qformer),
        "num_query_tokens":  args.num_query_tokens,
        "qformer_layers":    args.qformer_layers,
        "num_heads":         args.num_heads,
        "dynamic_cross_attn": _state_dict_to_cpu(dynamic_cross_attn),
        "mlp":               _state_dict_to_cpu(mlp_model),
        "input_dim":         unwrapped_mlp.mlp[0].in_features,
        "hidden_dim":        unwrapped_mlp.mlp[0].out_features,
        "output_dim":        unwrapped_mlp.mlp[-1].out_features,
        "emb_dim":           unwrapped_qformer.query_tokens.shape[-1],
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if pq_codebook_model is not None and getattr(args, 'finetune_codebook', False):
        pqm = _safe_unwrap(pq_codebook_model)
        ckpt["codebooks"]     = [cb.detach().cpu() for cb in pqm.codebooks]
        ckpt["quant_type"]    = pqm.quant_type
        if pqm.quant_type in {"pq", "opq"}:
            ckpt["num_subspaces"] = pqm.num_subspaces
            ckpt["subspace_dim"]  = pqm.subspace_dim
        if pqm.quant_type == "rq":
            ckpt["num_stages"] = pqm.num_stages
        if pqm.quant_type == "opq":
            ckpt["rotation"] = pqm.rotation.detach().cpu()
        ckpt["finetuned_codebook"] = True
    torch.save(ckpt, os.path.join(path, "model.pt"))
    print(f"Saved {'final' if final else 'checkpoint'} to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path",       required=True)
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--llm_path",        required=True)
    parser.add_argument("--codebook_path",   required=True)
    parser.add_argument("--encoder_path",    required=True)
    parser.add_argument("--output_dir",      required=True)

    parser.add_argument("--num_query_tokens", type=int, default=4)
    parser.add_argument("--qformer_layers",   type=int, default=2)
    parser.add_argument("--num_heads",        type=int, default=4)
    parser.add_argument("--hidden_dim",       type=int, default=None)
    parser.add_argument("--max_profile_len",  type=int, default=None)
    parser.add_argument("--max_length",       type=int, default=2048)

    parser.add_argument("--batch_size",                  type=int,   default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=8)
    parser.add_argument("--num_epochs",                  type=int,   default=3)
    parser.add_argument("--learning_rate",               type=float, default=1e-4)
    parser.add_argument("--weight_decay",                type=float, default=0.01)
    parser.add_argument("--max_grad_norm",               type=float, default=1.0)
    parser.add_argument("--save_steps",                  type=int,   default=1000)
    parser.add_argument("--seed",                        type=int,   default=42)
    parser.add_argument("--task",                        type=str,   default=None)
    parser.add_argument("--ablation",                    type=str,   default='both',
                        choices=['both', 'stable_only', 'dynamic_only'])
    parser.add_argument("--finetune_codebook", action="store_true",
                        help="Unfreeze the PQ codebook and train it during stage 2 (rebuttal ablation).")
    parser.add_argument(
        "--allow_embedding_dim_mismatch",
        action="store_true",
        help="Legacy compatibility: truncate/pad profile embeddings instead of failing on dimension mismatch.",
    )

    args = parser.parse_args()
    train(args)
