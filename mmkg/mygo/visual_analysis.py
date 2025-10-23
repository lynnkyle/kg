#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot high-D embeddings in 2D and judge closeness by original-space distances/similarities.
- Projection: PCA (NumPy SVD, no sklearn dependency)
- Similarity: cosine or euclidean (computed in ORIGINAL space)
- Visualization: 2D scatter + kNN edges from original space

Usage:
  python plot_embeddings_2d.py             # 运行内置演示数据
  python plot_embeddings_2d.py --npy my.npy --metric cosine --k 5 --threshold 0.75 --out fig.png
  python plot_embeddings_2d.py --csv my.csv --metric euclidean --k 10

Input:
  .npy: shape [n_samples, n_dims]
  .csv: rows=samples, cols=dims (no header)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---------- Core math ----------

def l2_normalize(x: np.ndarray, axis=1, eps=1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)

def pca_2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    # economy SVD
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T
    return Z

def pairwise_similarity(X: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """
    Return a matrix S where 'larger = more similar'.
    - cosine: in [-1, 1]
    - euclidean: we return negative Euclidean distance so that larger means closer
    """
    metric = metric.lower()
    if metric == "cosine":
        Xn = l2_normalize(X, axis=1)
        return Xn @ Xn.T
    elif metric == "euclidean":
        G = X @ X.T
        n = np.sum(X * X, axis=1, keepdims=True)
        D2 = n + n.T - 2 * G
        D2 = np.maximum(D2, 0.0)
        D = np.sqrt(D2)
        return -D
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")

def build_knn_pairs(S: np.ndarray, k: int = 5, threshold= None) -> list[tuple[int, int]]:
    """
    Build undirected edges from similarity matrix S (larger = closer).
    Only keep top-k neighbors per node; optionally apply a similarity threshold.
    """
    n = S.shape[0]
    pairs = set()
    for i in range(n):
        idx = np.argsort(-S[i])  # descending
        idx = idx[idx != i]
        taken = 0
        for j in idx:
            if threshold is not None and S[i, j] < threshold:
                continue
            a, b = (i, j) if i < j else (j, i)
            pairs.add((a, b))
            taken += 1
            if taken >= k:
                break
    return sorted(pairs)


# ---------- Plotting ----------

def plot_2d_with_edges(Z2: np.ndarray,
                       pairs: list[tuple[int, int]],
                       labels: np.ndarray | None = None,
                       title: str = "",
                       save_path = None) -> None:
    plt.figure(figsize=(8, 6))
    # draw edges first (light, behind points)
    for a, b in pairs:
        xa, ya = Z2[a, 0], Z2[a, 1]
        xb, yb = Z2[b, 0], Z2[b, 1]
        plt.plot([xa, xb], [ya, yb], linewidth=0.6, alpha=0.25)

    # scatter points
    if labels is None:
        plt.scatter(Z2[:, 0], Z2[:, 1], alpha=0.9)
    else:
        labels = np.asarray(labels)
        uniq = np.unique(labels)
        markers = ['o', 's', '^', 'v', 'D', 'P', 'X']
        for i, lab in enumerate(uniq):
            m = markers[i % len(markers)]
            idx = np.where(labels == lab)[0]
            plt.scatter(Z2[idx, 0], Z2[idx, 1], marker=m, alpha=0.9, label=str(lab))
        plt.legend(loc="best", frameon=True)

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180)
    plt.show()


# ---------- I/O helpers ----------

def load_embeddings(npy: str, csv: str) -> np.ndarray:
    if npy:
        return np.load(npy)
    if csv:
        return np.loadtxt(csv, delimiter=",")
    # fallback: demo data (3 clusters in 64D)
    rng = np.random.default_rng(0)
    n_per, d = 70, 64
    centers = np.stack([
        rng.normal(0, 1, d),
        rng.normal(0, 1, d) + 4,
        rng.normal(0, 1, d) - 4,
    ])
    X = np.vstack([
        centers[0] + rng.normal(0, 0.7, (n_per, d)),
        centers[1] + rng.normal(0, 0.7, (n_per, d)),
        centers[2] + rng.normal(0, 0.7, (n_per, d)),
    ])
    return X

def maybe_load_labels(path: str, n: int = None) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"labels file not found: {path}")
    # labels as one value per line (int or str)
    vals = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip() != ""]
    arr = np.array(vals)
    if n is not None and len(arr) != n:
        raise ValueError(f"labels length {len(arr)} != embeddings n {n}")
    return arr


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="2D plot of high-D embeddings with original-space kNN edges")
    ap.add_argument("--npy", type=str, default=None, help="path to .npy embeddings (shape [n,d])")
    ap.add_argument("--csv", type=str, default=None, help="path to .csv embeddings (rows=samples, cols=dims)")
    ap.add_argument("--labels", type=str, default=None, help="optional labels file (one label per line)")
    ap.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"], help="similarity metric")
    ap.add_argument("--k", type=int, default=5, help="k-NN edges per node (original space)")
    ap.add_argument("--threshold", type=float, default=None, help="similarity threshold (e.g., 0.75 for cosine)")
    ap.add_argument("--out", type=str, default=None, help="save figure to this path (e.g., fig.png)")
    args = ap.parse_args()

    X = load_embeddings(args.npy, args.csv)
    n = X.shape[0]
    y = maybe_load_labels(args.labels, n=n)

    # 1) Project to 2D for visualization
    Z2 = pca_2d(X)

    # 2) Original-space similarity
    S = pairwise_similarity(X, metric=args.metric)

    # 3) Build kNN edges (from original space)
    pairs = build_knn_pairs(S, k=args.k, threshold=args.threshold)

    # 4) Plot
    ttl = f"2D (PCA) + {args.metric} kNN edges (k={args.k}"
    if args.threshold is not None:
        ttl += f", thr={args.threshold}"
    ttl += ")"
    plot_2d_with_edges(Z2, pairs, labels=y, title=ttl, save_path=args.out)


if __name__ == "__main__":
    main()