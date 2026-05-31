"""Measure the intrinsic dimensionality of evo2 layer-22 activations.

Loads one shard (~100k tokens x 1920 dim) and computes:
  - SVD spectrum (singular values of centered data)
  - cumulative-variance curve: how many components for 50/90/95/99% variance
  - participation ratio: (sum lambda)^2 / sum(lambda^2)
  - stable rank: ||X||_F^2 / ||X||_op^2
"""
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

SHARD = "/data/interp/evo2/activations/evo2_1b_base_layer22_parquet_25M_prokeuk/shard_00000.parquet"
HIDDEN = 1920

print(f"loading {SHARD}")
t = pq.read_table(SHARD)
X = np.stack([t.column(f"dim_{i}").to_numpy() for i in range(HIDDEN)], axis=1).astype(np.float32)
print(f"X shape = {X.shape}, dtype = {X.dtype}")
print(f"per-token L2 norm: mean={np.linalg.norm(X, axis=1).mean():.2f}, std={np.linalg.norm(X, axis=1).std():.2f}")
print(f"per-dim mean: |mean(mean(X,axis=0))| = {np.abs(X.mean(0)).mean():.4f}, max = {np.abs(X.mean(0)).max():.4f}")
print()

print("centering by per-channel mean, computing SVD...")
Xc = X - X.mean(axis=0, keepdims=True)
# We only need singular values, not vectors. Use full_matrices=False for speed.
S = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
lam = (S ** 2) / (Xc.shape[0] - 1)   # eigenvalues of covariance (PCA variances)
total = lam.sum()
cum = np.cumsum(lam) / total
print(f"top-10 singular values: {S[:10].round(2)}")
print(f"top-10 eigenvalues  (lambda_i): {lam[:10].round(2)}")
print(f"total variance Sigma lambda = {total:.2f}")
print()

print(f"--- cumulative variance thresholds ---")
for thresh in [0.50, 0.80, 0.90, 0.95, 0.99, 0.999, 0.9999]:
    k = int(np.searchsorted(cum, thresh) + 1)
    print(f"  {thresh*100:5.2f}% variance:  k = {k:4d}  ({100*k/HIDDEN:5.2f}% of {HIDDEN} dims)")

print()
print(f"--- effective dimensionality scalars ---")
PR_linear = (lam.sum() ** 2) / (lam ** 2).sum()
stable_rank = (S ** 2).sum() / (S[0] ** 2)
print(f"  Participation Ratio (PR) = (Sigma lambda)^2 / Sigma lambda^2 = {PR_linear:.2f}")
print(f"     (1 = perfectly rank-1; HIDDEN = uniform spectrum)")
print(f"  Stable rank ||X||_F^2 / ||X||_op^2                          = {stable_rank:.2f}")
print(f"  Median entropy of normalized lambda                          = {(-(lam/total) * np.log2(np.where(lam>0, lam/total, 1))).sum():.2f} bits  (max = {np.log2(HIDDEN):.2f})")
print(f"  Effective dim via exp(entropy)                                = {2**(-(lam/total) * np.log2(np.where(lam>0, lam/total, 1))).sum():.2f}")

print()
print(f"--- interpretation ---")
k50 = int(np.searchsorted(cum, 0.50) + 1)
k90 = int(np.searchsorted(cum, 0.90) + 1)
k99 = int(np.searchsorted(cum, 0.99) + 1)
print(f"  Layer-22 activations have a 1920-dim ambient space, but functionally span only ~{int(PR_linear)} dims.")
print(f"  Half the variance lives in the top {k50} principal components ({100*k50/HIDDEN:.1f}% of dims).")
print(f"  99% of the variance lives in the top {k99} principal components ({100*k99/HIDDEN:.1f}% of dims).")
print(f"  This is the 'low-rank manifold' that the SAE's TopK k=32 can easily span.")

# Save spectrum to disk for downstream plotting
np.savez("/tmp/dim_analysis_layer22.npz", S=S, lam=lam, cum=cum, X_mean=X.mean(0), X_norms=np.linalg.norm(X, axis=1))
print(f"\nSpectrum saved to /tmp/dim_analysis_layer22.npz")
