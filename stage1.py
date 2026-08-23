import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
import argparse
import math
from typing import List, Dict, Any
import numpy as np
from collections import defaultdict


def _load_embedding_tensor(path):
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, torch.Tensor):
        embeddings = raw
    elif isinstance(raw, list):
        embeddings = torch.stack(raw)
    elif isinstance(raw, dict) and isinstance(raw.get("embeddings"), torch.Tensor):
        embeddings = raw["embeddings"]
    else:
        raise ValueError(f"Unknown embeddings format in {path}: {type(raw)}")
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError(f"Expected non-empty 2D embeddings, got {tuple(embeddings.shape)}")
    if not torch.is_floating_point(embeddings):
        raise ValueError(f"Embedding tensor must use a floating dtype, got {embeddings.dtype}")
    return embeddings.float().contiguous()


def _safe_unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _squared_distances(vectors, centroids):
    vectors_f = vectors.float()
    centroids_f = centroids.float()
    return (
        vectors_f.square().sum(dim=-1, keepdim=True)
        + centroids_f.square().sum(dim=-1).unsqueeze(0)
        - 2.0 * vectors_f @ centroids_f.t()
    ).clamp_min_(0)


def balanced_kmeans_init(embeddings, num_clusters, max_iters=100, device="cuda:0",
                         sinkhorn_iters=5):
    """Balanced soft k-means using Sinkhorn assignments.

    The previous implementation filled centroids one at a time, making cluster 0
    systematically choose before cluster 1 and requiring K separate sorts per
    iteration. Sinkhorn enforces equal soft mass for every centroid jointly.
    """
    if isinstance(device, str):
        device = torch.device(device)
    elif not isinstance(device, torch.device):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    embeddings = embeddings.to(device).float()
    N, emb_dim = embeddings.shape
    if N < num_clusters:
        raise ValueError(
            f"Balanced K-means needs at least as many samples as clusters: N={N}, K={num_clusters}"
        )
    
    print(f"Initializing codebook with Balanced K-means on {device}...")
    print(f"  Total embeddings: {N}")
    print(f"  Number of clusters: {num_clusters}")
    print(f"  Target soft mass per cluster: {N / num_clusters:.2f}")
    
    indices = torch.randperm(N, device=device)[:num_clusters]
    centroids = embeddings[indices].clone()
    
    previous_centroids = None
    for iteration in tqdm(range(max_iters), desc="Balanced K-means iterations"):
        distances = _squared_distances(embeddings, centroids)
        positive = distances[distances > 0]
        temperature = positive.median().clamp_min(1e-6) if positive.numel() else torch.tensor(1.0, device=device)
        scores = torch.exp((-distances / temperature).clamp(min=-50.0, max=0.0)).t()
        scores = scores / scores.sum().clamp_min(1e-12)
        for _ in range(sinkhorn_iters):
            scores = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
            scores = scores / num_clusters
            scores = scores / scores.sum(dim=0, keepdim=True).clamp_min(1e-12)
            scores = scores / N
        assignments = scores.t() * N
        centroids = assignments.t() @ embeddings
        centroids = centroids / assignments.sum(dim=0).unsqueeze(1).clamp_min(1e-12)

        if previous_centroids is not None:
            shift = (centroids - previous_centroids).square().mean().sqrt()
            if shift.item() < 1e-5:
                print(f"  Converged at iteration {iteration + 1} (centroid shift={shift.item():.2e})")
                break
        previous_centroids = centroids.clone()
    
    print(f"  Balanced K-means initialization completed!")
    return centroids


class CodebookModel(nn.Module):
    def __init__(self, codebook_size=10000, emb_dim=768, num_subspaces=4):
        super().__init__()
        if codebook_size < 2:
            raise ValueError("codebook_size must be at least 2")
        if num_subspaces < 1:
            raise ValueError("num_subspaces must be positive")
        self.codebook_size = codebook_size
        self.emb_dim = emb_dim
        self.num_subspaces = num_subspaces
        
        assert emb_dim % num_subspaces == 0, f"emb_dim ({emb_dim}) must be divisible by num_subspaces ({num_subspaces})"
        self.subspace_dim = emb_dim // num_subspaces
        
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, self.subspace_dim))
            for _ in range(num_subspaces)
        ])
        
        print(f"Initialized PQ codebook: {num_subspaces} subspaces, each {self.subspace_dim}D, {codebook_size} entries per subspace")
        print(f"  Total capacity: {codebook_size}^{num_subspaces} = {codebook_size ** num_subspaces:.2e} combinations")
    
    def quantize(self, embeddings):
        batch_size, seq_len, emb_dim = embeddings.shape
        if emb_dim != self.emb_dim:
            raise ValueError(f"Expected embedding dim {self.emb_dim}, got {emb_dim}")
        flat_embs = embeddings.reshape(-1, emb_dim)
        
        subspace_embs = flat_embs.reshape(-1, self.num_subspaces, self.subspace_dim)
        
        quantized_parts = []
        all_indices = []
        
        for i, codebook in enumerate(self.codebooks):
            subspace = subspace_embs[:, i, :]
            distances = _squared_distances(subspace, codebook)
            indices = torch.argmin(distances, dim=-1)
            quantized = codebook[indices]
            quantized_parts.append(quantized)
            all_indices.append(indices)
        
        quantized = torch.cat(quantized_parts, dim=-1)
        quantized = quantized.reshape(batch_size, seq_len, emb_dim)
        
        all_indices = torch.stack(all_indices, dim=-1)
        all_indices = all_indices.reshape(batch_size, seq_len, self.num_subspaces)
        
        return quantized, all_indices
    
    def forward(self, embeddings):
        quantized, indices = self.quantize(embeddings)
        return quantized, indices


class CodebookDataset(Dataset):
    def __init__(self, embeddings_path, batch_size_per_sample=8, device="cpu"):
        self.batch_size_per_sample = batch_size_per_sample
        self.device = device
        
        print(f"Loading embeddings from {embeddings_path}...")
        self.embeddings = _load_embedding_tensor(embeddings_path)
        
        print(f"Loaded {len(self.embeddings)} embeddings, shape: {self.embeddings.shape}")
        
        if self.embeddings.ndim != 2 or len(self.embeddings) == 0:
            raise ValueError(f"Embedding tensor must be non-empty and 2D, got {self.embeddings.shape}")
        self.num_samples = math.ceil(len(self.embeddings) / batch_size_per_sample)
        print(f"Creating {self.num_samples} samples (each with {batch_size_per_sample} embeddings)")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size_per_sample
        end_idx = min(start_idx + self.batch_size_per_sample, len(self.embeddings))
        
        sample_embs = self.embeddings[start_idx:end_idx]
        
        if len(sample_embs) < self.batch_size_per_sample:
            padding = sample_embs[-1:] if len(sample_embs) > 0 else torch.zeros(1, self.embeddings.shape[1])
            num_padding = self.batch_size_per_sample - len(sample_embs)
            sample_embs = torch.cat([sample_embs, padding.repeat(num_padding, 1)], dim=0)
        
        return {
            "embeddings": sample_embs[:self.batch_size_per_sample]
        }


class CodebookTrainer:
    def __init__(self, codebook_model, accelerator, 
                 codebook_weight=1.0, diversity_weight=0.1, usage_weight=0.3,
                 usage_temperature=0.1):
        self.codebook_model = codebook_model
        self.accelerator = accelerator
        self.codebook_weight = codebook_weight
        self.diversity_weight = diversity_weight
        self.usage_weight = usage_weight
        if usage_temperature <= 0:
            raise ValueError("usage_temperature must be positive")
        self.usage_temperature = usage_temperature
        self.base_diversity_weight = diversity_weight
        self.collapse_detected = False
        self.epoch_indices_list = []
        
    def compute_loss(self, batch):
        embeddings = batch["embeddings"]
        device = embeddings.device
        batch_size, batch_size_per_sample, emb_dim = embeddings.shape
        codebook_module = _safe_unwrap(self.codebook_model)
        
        quantized_embs, indices = self.codebook_model(embeddings)
        
        if not hasattr(self, '_diagnostic_printed'):
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if local_rank == 0:
                flat_embs = embeddings.view(-1, emb_dim)
                emb_std = flat_embs.std(dim=0).mean().item()
                emb_mean_dist = torch.cdist(flat_embs, flat_embs, p=2)
                mask = ~torch.eye(emb_mean_dist.size(0), dtype=torch.bool, device=emb_mean_dist.device)
                emb_mean_dist_value = emb_mean_dist[mask].mean().item() if mask.any() else 0.0
                
                print(f"\n[Initial Diagnostic - PQ]")
                print(f"  Embeddings std: {emb_std:.4f}")
                print(f"  Embeddings mean distance: {emb_mean_dist_value:.4f}")
                print(f"  Number of subspaces: {codebook_module.num_subspaces}")
                print(f"  Subspace dimension: {codebook_module.subspace_dim}")
            self._diagnostic_printed = True
        
        # Input embeddings are frozen offline data, so a VQ commitment term has
        # no trainable path. Keep it as a reconstruction diagnostic only.
        commitment_loss = F.mse_loss(quantized_embs.detach(), embeddings)
        codebook_loss = F.mse_loss(quantized_embs, embeddings.detach())
        total_codebook_loss = codebook_loss
        
        codebook_size = codebook_module.codebook_size
        num_subspaces = codebook_module.num_subspaces
        
        flat_indices = indices.view(-1, num_subspaces)

        # Differentiable batch-level code usage objective. The old unique-count
        # and entropy terms were based on argmin indices and therefore had zero
        # gradient almost everywhere.
        flat_embeddings = embeddings.reshape(-1, num_subspaces, codebook_module.subspace_dim)
        soft_usage_losses = []
        for subspace_index, codebook in enumerate(codebook_module.codebooks):
            distances = _squared_distances(
                flat_embeddings[:, subspace_index, :], codebook
            )
            assignment_probs = torch.softmax(
                -distances / self.usage_temperature, dim=-1
            )
            mean_usage = assignment_probs.mean(dim=0)
            normalized_entropy = -(
                mean_usage * torch.log(mean_usage.clamp_min(1e-12))
            ).sum() / math.log(codebook_size)
            soft_usage_losses.append(1.0 - normalized_entropy)
        differentiable_usage_loss = torch.stack(soft_usage_losses).mean()
        
        all_unique_counts = []
        all_usage_counts = []
        all_codebooks = []
        
        for i in range(num_subspaces):
            subspace_indices = flat_indices[:, i]
            unique_indices = torch.unique(subspace_indices)
            all_unique_counts.append(len(unique_indices))
            
            usage_counts = torch.zeros(codebook_size, device=indices.device, dtype=torch.long)
            unique_vals, counts = torch.unique(subspace_indices, return_counts=True)
            usage_counts[unique_vals] = counts
            all_usage_counts.append(usage_counts)
            
            all_codebooks.append(codebook_module.codebooks[i])
        
        all_unique_counts_tensor = torch.tensor(all_unique_counts, device=device, dtype=torch.float)
        avg_unique_per_subspace = all_unique_counts_tensor.mean()
        
        all_usage_counts_merged = sum(all_usage_counts)
        total_usage = all_usage_counts_merged.sum().float()
        usage_freq = all_usage_counts_merged.float() / (total_usage + 1e-10)
        
        num_unique_total = (all_usage_counts_merged > 0).sum().item()
        
        if avg_unique_per_subspace > 1:
            all_min_distances = []
            for i in range(num_subspaces):
                sub_unique_indices = torch.unique(flat_indices[:, i])
                if len(sub_unique_indices) > 0:
                    used_codebook = all_codebooks[i][sub_unique_indices]
                    
                    if used_codebook.size(0) > 1:
                        distances = torch.cdist(used_codebook, used_codebook, p=2)
                        mask = ~torch.eye(distances.size(0), dtype=torch.bool, device=distances.device)
                        if mask.any():
                            min_dist = distances[mask].min()
                            all_min_distances.append(min_dist)
            
            if len(all_min_distances) > 0:
                min_distance = torch.stack(all_min_distances).mean()
            else:
                min_distance = torch.tensor(0.0, device=indices.device)
            
            scale = 5.0
            diversity_loss = -scale * torch.tanh(min_distance / scale)
            
            num_used = (all_usage_counts_merged > 0).sum().float()
            usage_ratio = num_used / codebook_size
            
            subspace_usage_ratios = []
            for i in range(num_subspaces):
                sub_usage = (all_usage_counts[i] > 0).sum().float()
                sub_ratio = sub_usage / codebook_size
                subspace_usage_ratios.append(sub_ratio)
            
            avg_subspace_usage = torch.stack(subspace_usage_ratios).mean()
            
            unique_combinations = set()
            flat_indices_cpu = flat_indices.cpu().numpy()
            for idx_tuple in flat_indices_cpu:
                combo = tuple(idx_tuple)
                unique_combinations.add(combo)
            num_unique_combos = len(unique_combinations)
            max_combos = codebook_size ** num_subspaces
            total_samples = batch_size * batch_size_per_sample
            combo_ratio = torch.tensor(num_unique_combos / min(max_combos, total_samples), device=device)
            
            overall_usage_loss = 1.0 - usage_ratio
            subspace_usage_loss = 1.0 - avg_subspace_usage
            combo_diversity_loss = 1.0 - combo_ratio
            
            usage_ratio_loss = 0.4 * overall_usage_loss + 0.3 * subspace_usage_loss + 0.3 * combo_diversity_loss
            
            non_zero_mask = usage_freq > 1e-10
            if non_zero_mask.sum() > 0:
                entropy = -(usage_freq[non_zero_mask] * torch.log(usage_freq[non_zero_mask] + 1e-10)).sum()
                max_entropy = torch.log(torch.tensor(float(codebook_size), device=device))
                normalized_entropy = entropy / (max_entropy + 1e-10)
                entropy_loss = 1.0 - normalized_entropy
            else:
                entropy_loss = torch.tensor(1.0, device=device)
            
            usage_loss = 0.7 * usage_ratio_loss + 0.3 * entropy_loss
            
            if not hasattr(self, '_step_count'):
                self._step_count = 0
            self._step_count += 1
            
            if self._step_count % 100 == 0:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                if local_rank == 0:
                    total_unique_ratio = num_unique_total / codebook_size
                    used_codebook_ratio = (all_usage_counts_merged > 0).sum().item() / codebook_size
                    top10_usage = all_usage_counts_merged.topk(min(10, codebook_size)).values.float() / total_usage if total_usage > 0 else torch.zeros(10)
                    
                    if len(all_min_distances) > 0:
                        mean_distance = torch.stack(all_min_distances).mean().item()
                        max_distance = torch.stack(all_min_distances).max().item()
                    else:
                        mean_distance = 0.0
                        max_distance = 0.0
                    
                    entropy_val = -(usage_freq[non_zero_mask] * torch.log(usage_freq[non_zero_mask] + 1e-10)).sum().item() if non_zero_mask.sum() > 0 else 0.0
                    max_entropy_val = torch.log(torch.tensor(float(codebook_size), device=device)).item()
                    normalized_entropy_val = entropy_val / (max_entropy_val + 1e-10) if max_entropy_val > 0 else 0.0
                    
                    subspace_usage_info = []
                    for i in range(num_subspaces):
                        sub_unique = all_unique_counts[i]
                        sub_ratio = sub_unique / codebook_size
                        subspace_usage_info.append(f"Sub{i}: {sub_unique}/{codebook_size} ({sub_ratio*100:.1f}%)")
                    
                    unique_combinations_diag = set()
                    flat_indices_diag = indices.view(-1, num_subspaces).cpu().numpy()
                    for idx_tuple in flat_indices_diag:
                        combo = tuple(idx_tuple)
                        unique_combinations_diag.add(combo)
                    num_unique_combos_diag = len(unique_combinations_diag)
                    total_samples_in_batch = batch_size * batch_size_per_sample
                    max_possible_combos = codebook_size ** num_subspaces
                    combo_ratio_diag = num_unique_combos_diag / total_samples_in_batch
                    combo_coverage = num_unique_combos_diag / min(max_possible_combos, total_samples_in_batch)
                    
                    print(f"\n[Step {self._step_count}]")
                    print(f"  Total unique indices (across all subspaces): {num_unique_total} / {codebook_size} ({total_unique_ratio*100:.2f}%)")
                    print(f"  Avg unique per subspace: {avg_unique_per_subspace.item():.1f} / {codebook_size}")
                    print(f"  Per subspace: {', '.join(subspace_usage_info)}")
                    print(f"  Unique combinations: {num_unique_combos_diag} / {total_samples_in_batch} samples ({combo_ratio_diag*100:.2f}% unique)")
                    print(f"    (Each embedding is quantized to 4 indices, e.g., (207,216,35,901). This counts how many different 4-tuples are used.)")
                    print(f"    (Max possible combinations: {max_possible_combos:.2e}, Coverage: {combo_coverage*100:.2f}%)")
                    print(f"  Min distance: {min_distance.item():.4f}")
                    print(f"  Mean distance: {mean_distance:.4f}")
                    print(f"  Max distance: {max_distance:.4f}")
                    print(f"  Codebook loss: {total_codebook_loss.item():.6f} (commitment: {commitment_loss.item():.6f}, codebook: {codebook_loss.item():.6f})")
                    print(f"  Diversity loss: {diversity_loss.item():.4f}")
                    print(f"  Usage loss: {usage_loss.item():.4f} (overall: {overall_usage_loss.item():.4f}, subspace: {subspace_usage_loss.item():.4f}, combo: {combo_diversity_loss.item():.4f}, entropy: {entropy_loss.item():.4f})")
                    print(f"  Overall usage ratio: {usage_ratio.item():.4f} ({num_used.item():.0f}/{codebook_size})")
                    print(f"  Avg subspace usage: {avg_subspace_usage.item():.4f}")
                    print(f"  Combination diversity: {combo_ratio.item():.4f}")
                    print(f"  Entropy: {entropy_val:.4f} / {max_entropy_val:.4f} (normalized: {normalized_entropy_val:.4f})")
                    print(f"  Top-10 usage: {top10_usage.tolist()}")
                    
                    if avg_unique_per_subspace < codebook_size * 0.01:
                        print(f"  ⚠️  WARNING: Codebook collapse! Only {avg_unique_per_subspace.item():.1f} unique indices per subspace on average.")
        elif avg_unique_per_subspace <= 1:
            diversity_loss = torch.tensor(10.0, device=indices.device)
            usage_loss = torch.tensor(1.0, device=indices.device)
            
            if not self.collapse_detected:
                self.collapse_detected = True
                self.diversity_weight = self.base_diversity_weight * 5.0
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                if local_rank == 0:
                    flat_embs = embeddings.view(-1, emb_dim)
                    emb_std = flat_embs.std(dim=0).mean().item()
                    emb_mean_dist = torch.cdist(flat_embs, flat_embs, p=2)
                    mask = ~torch.eye(emb_mean_dist.size(0), dtype=torch.bool, device=emb_mean_dist.device)
                    emb_mean_dist_value = emb_mean_dist[mask].mean().item() if mask.any() else 0.0
                    
                    print(f"\n⚠️  Codebook collapse detected! Only 1 entry used.")
                    print(f"  Embeddings std: {emb_std:.4f}")
                    print(f"  Embeddings mean distance: {emb_mean_dist_value:.4f}")
                    print(f"  Increasing diversity_weight to {self.diversity_weight}")
                    print(f"  ⚠️  If embeddings are too similar, codebook may collapse!")
        else:
            diversity_loss = torch.tensor(10.0, device=indices.device)
            usage_loss = torch.tensor(1.0, device=indices.device)
        
        usage_loss = differentiable_usage_loss
        total_loss = (self.codebook_weight * total_codebook_loss + 
                     self.diversity_weight * diversity_loss +
                     self.usage_weight * usage_loss)
        
        if hasattr(self, '_collecting_epoch_stats') and self._collecting_epoch_stats:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if local_rank == 0:
                indices_cpu = indices.detach().cpu().numpy()
                self.epoch_indices_list.append(indices_cpu)
        
        return total_loss, total_codebook_loss, diversity_loss, usage_loss, all_unique_counts
    
    def start_epoch_stats(self):
        self._collecting_epoch_stats = True
        self.epoch_indices_list = []
    
    def end_epoch_stats(self):
        if not hasattr(self, '_collecting_epoch_stats') or not self._collecting_epoch_stats:
            return None
        
        if len(self.epoch_indices_list) == 0:
            self._collecting_epoch_stats = False
            return None
        
        all_indices = np.concatenate(self.epoch_indices_list, axis=0)
        codebook_module = _safe_unwrap(self.codebook_model)
        flat_indices = all_indices.reshape(-1, codebook_module.num_subspaces)
        
        codebook_size = codebook_module.codebook_size
        num_subspaces = codebook_module.num_subspaces
        
        subspace_stats = []
        all_used_indices_set = set()
        
        for subspace_idx in range(num_subspaces):
            subspace_indices = flat_indices[:, subspace_idx]
            unique_subspace_indices = np.unique(subspace_indices)
            num_unique_subspace = len(unique_subspace_indices)
            usage_ratio_subspace = num_unique_subspace / codebook_size
            
            all_used_indices_set.update(unique_subspace_indices.tolist())
            
            subspace_stats.append({
                'subspace_idx': subspace_idx,
                'num_unique': num_unique_subspace,
                'usage_ratio': usage_ratio_subspace
            })
        
        total_unique_entries = len(all_used_indices_set)
        total_possible_entries = codebook_size
        total_usage_ratio = total_unique_entries / total_possible_entries if total_possible_entries > 0 else 0.0
        
        unique_combinations = set()
        for idx_tuple in flat_indices:
            combo = tuple(idx_tuple)
            unique_combinations.add(combo)
        num_unique_combos = len(unique_combinations)
        max_possible_combos = min(codebook_size ** num_subspaces, len(flat_indices))
        combo_ratio = num_unique_combos / max_possible_combos if max_possible_combos > 0 else 0.0
        
        self.epoch_indices_list = []
        self._collecting_epoch_stats = False
        
        return {
            'subspace_stats': subspace_stats,
            'total_unique_entries': total_unique_entries,
            'total_possible_entries': total_possible_entries,
            'total_usage_ratio': total_usage_ratio,
            'num_unique_combos': num_unique_combos,
            'max_possible_combos': max_possible_combos,
            'combo_ratio': combo_ratio,
            'total_samples': len(flat_indices)
        }


def train(args):
    set_seed(args.seed, device_specific=True)
    accelerator = Accelerator()
    embedding_probe = _load_embedding_tensor(args.embeddings_path)
    inferred_emb_dim = int(embedding_probe.shape[1])
    if args.emb_dim is None:
        args.emb_dim = inferred_emb_dim
    elif args.emb_dim != inferred_emb_dim:
        raise ValueError(
            f"--emb_dim={args.emb_dim} does not match embedding file dimension {inferred_emb_dim}"
        )
    
    if accelerator.is_local_main_process:
        print(f"Stage 1: Training codebook only (without LLM)")
        print(f"Codebook size: {args.codebook_size}")
        print(f"Codebook dim: {args.emb_dim} ({args.num_subspaces} subspaces x {args.emb_dim // args.num_subspaces}D)")
        print(f"Batch size per sample: {args.batch_size_per_sample}")
    
    codebook_model = CodebookModel(
        codebook_size=args.codebook_size,
        emb_dim=args.emb_dim,
        num_subspaces=args.num_subspaces
    )
    
    if args.use_balanced_kmeans_init:
        init_codebook_path = os.path.join(
            args.output_dir, 
            f"codebook_init_size{args.codebook_size}_dim{args.emb_dim}_pq{args.num_subspaces}_sample{args.init_sample_size if args.init_sample_size else 'all'}.pt"
        )
        
        if os.path.exists(init_codebook_path):
            if accelerator.is_local_main_process:
                print(f"Loading saved PQ initialization from {init_codebook_path}...")
            init_data = torch.load(init_codebook_path, map_location='cpu')
            
            if isinstance(init_data, dict) and 'codebooks' in init_data:
                init_codebooks = init_data['codebooks']
                if len(init_codebooks) == codebook_model.num_subspaces:
                    for i, init_cb in enumerate(init_codebooks):
                        if init_cb.shape == (args.codebook_size, codebook_model.subspace_dim):
                            codebook_model.codebooks[i].data = init_cb
                        else:
                            raise ValueError(f"Subspace {i} shape mismatch! Expected ({args.codebook_size}, {codebook_model.subspace_dim}), got {init_cb.shape}")
                    if accelerator.is_local_main_process:
                        print("✅ Loaded saved PQ initialization!")
                else:
                    raise ValueError(f"Number of subspaces mismatch! Expected {codebook_model.num_subspaces}, got {len(init_codebooks)}")
            elif isinstance(init_data, torch.Tensor):
                if init_data.shape == (args.codebook_size, args.emb_dim):
                    if accelerator.is_local_main_process:
                        print("Converting old format to PQ format...")
                    for i in range(codebook_model.num_subspaces):
                        start_idx = i * codebook_model.subspace_dim
                        end_idx = (i + 1) * codebook_model.subspace_dim
                        codebook_model.codebooks[i].data = init_data[:, start_idx:end_idx]
                    if accelerator.is_local_main_process:
                        print("✅ Converted and loaded initialization!")
                else:
                    raise ValueError(f"Shape mismatch! Expected ({args.codebook_size}, {args.emb_dim}), got {init_data.shape}")
            else:
                raise ValueError(f"Unknown initialization format: {type(init_data)}")
            
            accelerator.wait_for_everyone()
        else:
            if accelerator.is_local_main_process:
                print("Initializing codebook with Balanced K-means...")
            
            temp_codebook_path = os.path.join(args.output_dir, "temp_codebook_init.pt")
            
            if accelerator.is_local_main_process:
                if args.embeddings_path and os.path.exists(args.embeddings_path):
                    print(f"Loading pre-computed embeddings from {args.embeddings_path}...")
                    init_embeddings = embedding_probe
                    
                    print(f"Loaded {len(init_embeddings)} pre-computed embeddings")
                    
                    if args.init_sample_size and len(init_embeddings) > args.init_sample_size:
                        print(f"Sampling {args.init_sample_size} embeddings...")
                        sample_indices = torch.randperm(len(init_embeddings))[:args.init_sample_size]
                        init_embeddings = init_embeddings[sample_indices]
                else:
                    print("No pre-computed embeddings provided, encoding from JSONL...")
                    if not args.encoder_path:
                        raise ValueError("--encoder_path is required when --embeddings_path file does not exist")
                    if not args.data_path:
                        raise ValueError("--data_path is required when --embeddings_path file does not exist")
                    print(f"Loading encoder from {args.encoder_path}...")
                    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_path)
                    encoder_model = AutoModel.from_pretrained(args.encoder_path)
                    encoder_model.eval()
                    
                    print("Reading data and encoding embeddings for initialization...")
                    init_embeddings_list = []
                    with open(args.data_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if args.init_sample_size and len(init_embeddings_list) >= args.init_sample_size:
                                break
                            item = json.loads(line.strip())
                            user_content = item.get("User-Generated Content", [])
                            for content_item in user_content:
                                if args.init_sample_size and len(init_embeddings_list) >= args.init_sample_size:
                                    break
                                prompt_text = content_item.get("prompt", "").strip()
                                comment_text = content_item.get("comment", "").strip()
                                if prompt_text and comment_text:
                                    text = f"Question: {prompt_text} Answer: {comment_text}"
                                    with torch.no_grad():
                                        encoder_device = next(encoder_model.parameters()).device
                                        inputs = encoder_tokenizer(
                                            text,
                                            padding=True,
                                            truncation=True,
                                            max_length=512,
                                            return_tensors="pt"
                                        ).to(encoder_device)
                                        outputs = encoder_model(**inputs)
                                        embeddings = outputs.last_hidden_state
                                        attention_mask = inputs['attention_mask']
                                        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
                                        summed = torch.sum(masked_embeddings, dim=1)
                                        summed_mask = torch.clamp(torch.sum(attention_mask, dim=1, keepdim=True), min=1e-9)
                                        mean_pooled = summed / summed_mask
                                        init_embeddings_list.append(mean_pooled.squeeze(0).cpu())
                    
                    init_embeddings = torch.stack(init_embeddings_list)
                    print(f"Encoded {len(init_embeddings)} embeddings for initialization")
                    
                    if args.init_sample_size and len(init_embeddings) > args.init_sample_size:
                        print(f"Sampling {args.init_sample_size} embeddings...")
                        sample_indices = torch.randperm(len(init_embeddings))[:args.init_sample_size]
                        init_embeddings = init_embeddings[sample_indices]
                
                init_device = "cuda:0" if torch.cuda.is_available() else "cpu"
                if init_embeddings.shape[1] != args.emb_dim:
                    raise ValueError(
                        f"Initialization embeddings have dim={init_embeddings.shape[1]}, expected {args.emb_dim}"
                    )
                print(f"Using embeddings directly ({args.emb_dim} dim) for PQ initialization...")
                
                print(f"Initializing {codebook_model.num_subspaces} subspaces independently...")
                initialized_codebooks = []
                
                for i in range(codebook_model.num_subspaces):
                    start_idx = i * codebook_model.subspace_dim
                    end_idx = (i + 1) * codebook_model.subspace_dim
                    subspace_embs = init_embeddings[:, start_idx:end_idx]
                    
                    if args.init_sample_size and len(subspace_embs) > args.init_sample_size:
                        sample_indices = torch.randperm(len(subspace_embs), generator=torch.Generator().manual_seed(i))[:args.init_sample_size]
                        subspace_embs = subspace_embs[sample_indices]
                    
                    print(f"  Initializing subspace {i+1}/{codebook_model.num_subspaces} (dim {start_idx}:{end_idx}) with {len(subspace_embs)} samples...")
                    initialized_subspace = balanced_kmeans_init(
                        embeddings=subspace_embs,
                        num_clusters=args.codebook_size,
                        max_iters=args.kmeans_max_iters,
                        device=init_device
                    )
                    initialized_codebooks.append(initialized_subspace.cpu())
                    codebook_model.codebooks[i].data = initialized_subspace.cpu()
                
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save({
                    'codebooks': initialized_codebooks,
                    'num_subspaces': codebook_model.num_subspaces,
                    'subspace_dim': codebook_model.subspace_dim,
                    'emb_dim': args.emb_dim,
                    'quant_type': 'pq',
                }, init_codebook_path)
                print(f"✅ Saved PQ initialization to {init_codebook_path}")
                
                import shutil
                temp_file = temp_codebook_path + ".tmp"
                torch.save({
                    'codebooks': [cb.data.cpu() for cb in codebook_model.codebooks],
                    'num_subspaces': codebook_model.num_subspaces,
                    'subspace_dim': codebook_model.subspace_dim,
                    'emb_dim': args.emb_dim,
                    'quant_type': 'pq',
                }, temp_file)
                shutil.move(temp_file, temp_codebook_path)
                print(f"Saved temporary PQ file for other processes")
            else:
                import time
                max_wait_time = 7200
                wait_interval = 10
                elapsed_time = 0
                
                while elapsed_time < max_wait_time:
                    if os.path.exists(temp_codebook_path):
                        try:
                            file_size = os.path.getsize(temp_codebook_path)
                            expected_size = codebook_model.num_subspaces * args.codebook_size * codebook_model.subspace_dim * 4
                            if file_size >= expected_size * 0.9:
                                init_data = torch.load(temp_codebook_path, map_location='cpu')
                                if isinstance(init_data, dict) and 'codebooks' in init_data:
                                    init_codebooks = init_data['codebooks']
                                    if len(init_codebooks) == codebook_model.num_subspaces:
                                        for i, init_cb in enumerate(init_codebooks):
                                            if init_cb.shape == (args.codebook_size, codebook_model.subspace_dim):
                                                codebook_model.codebooks[i].data = init_cb
                                            else:
                                                print(f"Shape mismatch for subspace {i}")
                                                break
                                        else:
                                            print("PQ codebook loaded!")
                                            break
                                elif isinstance(init_data, torch.Tensor):
                                    if init_data.shape == (args.codebook_size, args.emb_dim):
                                        for i in range(codebook_model.num_subspaces):
                                            start_idx = i * codebook_model.subspace_dim
                                            end_idx = (i + 1) * codebook_model.subspace_dim
                                            codebook_model.codebooks[i].data = init_data[:, start_idx:end_idx]
                                        print("Codebook converted and loaded!")
                                        break
                        except Exception as e:
                            print(f"File not ready: {e}")
                    
                    time.sleep(wait_interval)
                    elapsed_time += wait_interval
                    if elapsed_time % 60 == 0:
                        print(f"Waiting... ({elapsed_time//60}min)")
                
                if not os.path.exists(temp_codebook_path):
                    raise RuntimeError(f"Timeout after {max_wait_time//60}min")
            
            accelerator.wait_for_everyone()
            
            if accelerator.is_local_main_process and os.path.exists(temp_codebook_path):
                try:
                    os.remove(temp_codebook_path)
                except:
                    pass
    else:
        if accelerator.is_local_main_process:
            print("Using random initialization")

    del embedding_probe
    dataset = CodebookDataset(
        embeddings_path=args.embeddings_path,
        batch_size_per_sample=args.batch_size_per_sample,
        device="cpu"
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    trainer = CodebookTrainer(
        codebook_model=codebook_model,
        accelerator=accelerator,
        codebook_weight=args.codebook_weight,
        diversity_weight=args.diversity_weight,
        usage_weight=args.usage_weight,
        usage_temperature=args.usage_temperature,
    )
    
    optimizer = torch.optim.AdamW(
        codebook_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    codebook_model, optimizer, dataloader = accelerator.prepare(
        codebook_model, optimizer, dataloader
    )
    trainer.codebook_model = codebook_model
    
    codebook_model.train()
    global_step = 0
    
    for epoch in range(args.num_epochs):
        if accelerator.is_local_main_process:
            print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        if accelerator.is_local_main_process:
            trainer.start_epoch_stats()
        
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        
        for batch in progress_bar:
            optimizer.zero_grad(set_to_none=True)
            
            total_loss, codebook_loss, diversity_loss, usage_loss, unique_counts = trainer.compute_loss(batch)

            accelerator.backward(total_loss)

            if args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(codebook_model.parameters(), args.max_grad_norm)

            optimizer.step()
            global_step += 1

            if accelerator.is_local_main_process:
                K = args.codebook_size
                use_str = "/".join(f"{c}" for c in unique_counts)
                avg_use = sum(unique_counts) / len(unique_counts) if unique_counts else 0
                progress_bar.set_postfix({
                    "loss": f"{total_loss.item():.4f}",
                    "cb": f"{codebook_loss.item():.4f}",
                    "div": f"{diversity_loss.item():.3f}",
                    "usg": f"{usage_loss.item():.3f}",
                    "use": f"{use_str}/{K}",
                    "avg%": f"{100*avg_use/K:.1f}",
                })
            
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"stage1_checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    
                    unwrapped_model = _safe_unwrap(codebook_model)
                    checkpoint_dict = {
                        "codebooks": [cb.data.cpu() for cb in unwrapped_model.codebooks],
                        "num_subspaces": unwrapped_model.num_subspaces,
                        "subspace_dim": unwrapped_model.subspace_dim,
                        "step": global_step,
                        "codebook_size": args.codebook_size,
                        "emb_dim": unwrapped_model.emb_dim,
                        "seed": args.seed,
                    }
                    
                    torch.save(checkpoint_dict, os.path.join(save_path, "codebook_model.pt"))
                    print(f"Saved checkpoint to {save_path}")
        
        if accelerator.is_local_main_process:
            epoch_stats = trainer.end_epoch_stats()
            if epoch_stats is not None:
                unwrapped_model = _safe_unwrap(codebook_model)
                print(f"\n{'='*60}")
                print(f"Epoch {epoch+1}/{args.num_epochs} - Codebook Usage Statistics")
                print(f"{'='*60}")
                print(f"Total samples processed: {epoch_stats['total_samples']}")
                print(f"\nPer Subspace Usage:")
                for sub_stat in epoch_stats['subspace_stats']:
                    print(f"  Subspace {sub_stat['subspace_idx']}: {sub_stat['num_unique']} / {unwrapped_model.codebook_size} unique entries used ({sub_stat['usage_ratio']*100:.2f}%)")
                print(f"\nOverall Usage (across all subspaces, deduplicated):")
                print(f"  Total unique codes used: {epoch_stats['total_unique_entries']} / {epoch_stats['total_possible_entries']} ({epoch_stats['total_usage_ratio']*100:.2f}%)")
                print(f"\nCombination Diversity:")
                print(f"  Unique combinations: {epoch_stats['num_unique_combos']} / {epoch_stats['max_possible_combos']} ({epoch_stats['combo_ratio']*100:.2f}%)")
                print(f"{'='*60}\n")
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "stage1_final")
        os.makedirs(final_path, exist_ok=True)
        
        unwrapped_model = _safe_unwrap(codebook_model)
        final_dict = {
            "codebooks": [cb.data.cpu() for cb in unwrapped_model.codebooks],
            "num_subspaces": unwrapped_model.num_subspaces,
            "subspace_dim": unwrapped_model.subspace_dim,
            "codebook_size": args.codebook_size,
            "emb_dim": unwrapped_model.emb_dim,
            "quant_type": "pq",
            "format_version": 2,
            "seed": args.seed,
        }
        
        torch.save(final_dict, os.path.join(final_path, "codebook_model.pt"))
        print(f"Stage 1 training completed! Model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--encoder_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    
    parser.add_argument("--codebook_size", type=int, default=1000)
    parser.add_argument("--emb_dim", type=int, default=None,
                        help="Embedding dimension; inferred from --embeddings_path when omitted")
    parser.add_argument("--num_subspaces", type=int, default=4)
    parser.add_argument("--batch_size_per_sample", type=int, default=8)
    
    parser.add_argument("--codebook_weight", type=float, default=1.0)
    parser.add_argument("--diversity_weight", type=float, default=0.1)
    parser.add_argument("--usage_weight", type=float, default=0.3)
    parser.add_argument("--usage_temperature", type=float, default=0.1,
                        help="Soft-assignment temperature for differentiable code usage regularization")
    
    parser.add_argument("--use_balanced_kmeans_init", action="store_true")
    parser.add_argument("--init_sample_size", type=int, default=None)
    parser.add_argument("--kmeans_max_iters", type=int, default=100)
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    train(args)
