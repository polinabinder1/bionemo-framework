"""Random ACGT sequences for architecture-vs-content sink test.

Conditions:
  uniform   100 seqs, 8192 bp, P(A)=P(C)=P(G)=P(T)=0.25
  gc_rich   100 seqs, 8192 bp, P(G)=P(C)=0.30, P(A)=P(T)=0.20  (60% GC)
  at_rich   100 seqs, 8192 bp, P(G)=P(C)=0.20, P(A)=P(T)=0.30  (40% GC)
"""
import numpy as np

OUT = "/data/interp/evo2/scratch/random_acgt_test.fasta"
SEQ_LEN = 8192
N_PER = 100
rng = np.random.default_rng(7)

CONDS = [
    ("uniform", np.array([0.25, 0.25, 0.25, 0.25])),
    ("gc_rich", np.array([0.20, 0.30, 0.30, 0.20])),   # A, C, G, T
    ("at_rich", np.array([0.30, 0.20, 0.20, 0.30])),
]
BASES = np.array(list("ACGT"))

count = {}
with open(OUT, "w") as f:
    rec = 0
    for cond, p in CONDS:
        for i in range(N_PER):
            idx = rng.choice(4, size=SEQ_LEN, p=p)
            seq = "".join(BASES[idx])
            f.write(f">cond={cond}|src=random|src_idx={i}|tag_len=0|total_len={SEQ_LEN}\n{seq}\n")
            rec += 1
        count[cond] = N_PER

print(f"wrote {rec} records to {OUT}")
for k, v in count.items():
    print(f"  {k}: {v}")
# Sanity: actual GC content per condition
for cond, p in CONDS:
    expected = p[1] + p[2]
    print(f"  {cond}: expected GC = {expected:.2f}")
