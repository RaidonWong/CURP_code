"""Build RQ (residual) and OPQ (rotated PQ) codebooks for the CURP rebuttal
quantization-alternatives comparison (Claim 2 / reviewer Z794).

Output format is consumed directly by stage2.py's PQCodebookModel:
  - RQ : {"quant_type":"rq",  "codebooks":[stage0..stageS] (each K×D), "num_stages"}
  - OPQ: {"quant_type":"opq", "codebooks":[sub0..subM] (each K×d), "rotation": R(D×D),
          "num_subspaces", "subspace_dim"}
So stage-2 training / inference are IDENTICAL to the PQ pipeline; only the
quantizer differs. Codebooks are fit with k-means (frozen thereafter), mirroring
the paper's Stage-1 role (a global unsupervised codebook).
"""
import argparse
import torch


def load_embeddings(path, sample, device):
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and "embeddings" in raw:
        X = raw["embeddings"]
    elif isinstance(raw, list):
        X = torch.stack(raw)
    else:
        X = raw
    X = X.float()
    N = X.shape[0]
    if sample and N > sample:
        idx = torch.randperm(N)[:sample]
        X = X[idx]
    return X.to(device)


@torch.no_grad()
def _assign(X, C, chunk=20000):
    """Nearest-centroid indices, chunked to bound memory."""
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for i in range(0, X.shape[0], chunk):
        xb = X[i:i + chunk]
        d = torch.cdist(xb, C, p=2)
        out[i:i + chunk] = torch.argmin(d, dim=-1)
    return out


@torch.no_grad()
def kmeans(X, K, iters=25, seed_shift=0):
    """Simple GPU k-means. Deterministic-ish init by strided sampling."""
    N, D = X.shape
    # init: strided pick of K points (avoids Math.random; varies by seed_shift)
    step = max(1, N // K)
    init_idx = (torch.arange(K, device=X.device) * step + seed_shift) % N
    C = X[init_idx].clone()
    for _ in range(iters):
        assign = _assign(X, C)
        newC = C.clone()
        for k in range(K):
            mask = assign == k
            cnt = mask.sum()
            if cnt > 0:
                newC[k] = X[mask].mean(dim=0)
            else:
                # reseed empty cluster to a random-ish far point
                newC[k] = X[(k * 7919 + 1) % N]
        shift = (newC - C).norm()
        C = newC
        if shift < 1e-4:
            break
    return C


@torch.no_grad()
def build_rq(X, K, num_stages, iters):
    """Residual quantization: full-dim codebooks applied in stages."""
    codebooks = []
    residual = X.clone()
    for s in range(num_stages):
        C = kmeans(residual, K, iters=iters, seed_shift=s)
        codebooks.append(C.cpu())
        assign = _assign(residual, C)
        residual = residual - C[assign]
    recon_err = residual.norm(dim=-1).mean().item()
    return codebooks, recon_err


@torch.no_grad()
def build_opq(X, K, num_subspaces, iters):
    """Parametric OPQ: variance-balancing rotation (eigen) + per-subspace PQ."""
    N, D = X.shape
    d = D // num_subspaces
    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    # eigendecomposition of covariance
    cov = (Xc.t() @ Xc) / N
    evals, evecs = torch.linalg.eigh(cov)                 # ascending
    order = torch.argsort(evals, descending=True)
    evals = evals[order]; evecs = evecs[:, order]
    # greedy balanced allocation of eigen-dims to subspaces (balance log-variance)
    load = [0.0] * num_subspaces
    slot = [[] for _ in range(num_subspaces)]
    logv = torch.log(evals.clamp(min=1e-8))
    for j in range(D):
        s = min(range(num_subspaces), key=lambda s: (load[s], len(slot[s]) >= d))
        # pick least-loaded subspace that still has room
        cand = [s for s in range(num_subspaces) if len(slot[s]) < d]
        s = min(cand, key=lambda s: load[s])
        slot[s].append(j); load[s] += logv[j].item()
    perm = [j for s in range(num_subspaces) for j in slot[s]]
    R = evecs[:, perm]                                    # (D, D) orthonormal columns
    # Rotation comes from centered covariance eigenvectors, but codebooks are fit on
    # the UNCENTERED rotated data X@R so they match what stage-2 quantizes at run time
    # (PQCodebookModel does flat_embs @ R with no centering).
    Xr = X @ R
    codebooks = []
    total_err = 0.0
    for i in range(num_subspaces):
        sub = Xr[:, i * d:(i + 1) * d].contiguous()
        C = kmeans(sub, K, iters=iters, seed_shift=i)
        codebooks.append(C.cpu())
        assign = _assign(sub, C)
        total_err += (sub - C[assign]).norm(dim=-1).pow(2).sum().item()
    recon_err = (total_err / N) ** 0.5
    return codebooks, R.cpu(), recon_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_path", required=True)
    ap.add_argument("--method", choices=["rq", "opq"], required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--codebook_size", type=int, default=1000)
    ap.add_argument("--num_subspaces", type=int, default=4)   # opq
    ap.add_argument("--num_stages", type=int, default=4)      # rq
    ap.add_argument("--sample", type=int, default=200000)
    ap.add_argument("--iters", type=int, default=25)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embeddings {args.embeddings_path} (sample={args.sample}) on {device}...")
    X = load_embeddings(args.embeddings_path, args.sample, device)
    print(f"Fitting {args.method} on {tuple(X.shape)}")

    if args.method == "rq":
        codebooks, err = build_rq(X, args.codebook_size, args.num_stages, args.iters)
        ckpt = {
            "quant_type": "rq",
            "codebooks": codebooks,
            "num_stages": args.num_stages,
            "codebook_size": args.codebook_size,
            "emb_dim": codebooks[0].shape[1],
        }
    else:
        codebooks, R, err = build_opq(X, args.codebook_size, args.num_subspaces, args.iters)
        ckpt = {
            "quant_type": "opq",
            "codebooks": codebooks,
            "rotation": R,
            "num_subspaces": args.num_subspaces,
            "subspace_dim": codebooks[0].shape[1],
            "codebook_size": args.codebook_size,
            "emb_dim": R.shape[0],
        }

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(ckpt, args.output_path)
    print(f"[{args.method}] saved {args.output_path}  mean residual/recon err = {err:.4f}")


if __name__ == "__main__":
    main()
