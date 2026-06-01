"""Compose combined test FASTA with multiple conditions in one file.

Conditions (headers carry cond= for downstream slicing):
  control_full         100 sequences, 8192 bp, no tag                  (baseline)
  tagged_full          100 sequences, full GTDB/euk tag + DNA up to 8192 total
  tagged_partial        50 sequences, |d__X;p__None;c__None;...;s__None| + DNA
  tagged_minimal        50 sequences, |d__Bacteria| or |d__Eukaryota| + DNA
  length_4096          100 sequences, truncated to 4096 bp
  length_6144          100 sequences, truncated to 6144 bp
"""
import random
from pathlib import Path

SRC_FASTA = "/data/interp/evo2/scratch/mixed_25M_prokeuk_v2.fasta"
OUT_FASTA = "/data/interp/evo2/scratch/special_token_test.fasta"
PROK_MAX = 1500
random.seed(7)

# Three hardcoded full tags (mix prok and euk)
TAG_PROK_1 = "|d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia coli|"
TAG_PROK_2 = "|d__Bacteria;p__Bacillota;c__Bacilli;o__Bacillales;f__Bacillaceae;g__Bacillus;s__Bacillus subtilis|"
TAG_EUK    = "|d__Eukaryota;p__Chordata;c__Mammalia;o__Primates;f__Hominidae;g__Homo;s__Homo sapiens|"

TAG_PARTIAL_PROK = "|d__Bacteria;p__None;c__None;o__None;f__None;g__None;s__None|"
TAG_PARTIAL_EUK  = "|d__Eukaryota;p__None;c__None;o__None;f__None;g__None;s__None|"
TAG_MIN_PROK = "|d__Bacteria|"
TAG_MIN_EUK  = "|d__Eukaryota|"

print(f"tag lengths: full_prok1={len(TAG_PROK_1)} full_prok2={len(TAG_PROK_2)} full_euk={len(TAG_EUK)}"
      f" partial={len(TAG_PARTIAL_PROK)} minimal_prok={len(TAG_MIN_PROK)}")


def load_fasta(path):
    seqs, headers, cur = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur)); cur = []
                headers.append(line[1:].strip())
            else:
                cur.append(line.strip())
        if cur: seqs.append("".join(cur))
    return headers, seqs


print("loading source FASTA...")
hdrs, seqs = load_fasta(SRC_FASTA)
print(f"  {len(seqs)} sequences")
prok_idx = [i for i in range(len(seqs)) if i < PROK_MAX and len(seqs[i]) >= 8192]
euk_idx  = [i for i in range(len(seqs)) if i >= PROK_MAX and len(seqs[i]) >= 8192]
random.shuffle(prok_idx); random.shuffle(euk_idx)
print(f"  prok (>=8192): {len(prok_idx)}, euk (>=8192): {len(euk_idx)}")


def emit(out, cond, src_idx, src_kind, body, tag_len):
    out.write(f">cond={cond}|src={src_kind}|src_idx={src_idx}|tag_len={tag_len}|total_len={len(body)}\n")
    out.write(body + "\n")


count = {}
def inc(c): count[c] = count.get(c, 0) + 1


with open(OUT_FASTA, "w") as out:
    # ---- control_full (100): 50 prok + 50 euk, 8192 bp DNA, no tag ----
    for i in range(50):
        emit(out, "control_full", prok_idx[i], "prok", seqs[prok_idx[i]][:8192], 0); inc("control_full")
        emit(out, "control_full", euk_idx[i],  "euk",  seqs[euk_idx[i]][:8192],  0); inc("control_full")
    base_prok = 50; base_euk = 50

    # ---- tagged_full (100): 33 prok+TAG1, 33 prok+TAG2, 34 euk+TAG_EUK ----
    for i in range(33):
        tag = TAG_PROK_1
        dna = seqs[prok_idx[base_prok + i]][:8192 - len(tag)]
        emit(out, "tagged_full", prok_idx[base_prok+i], "prok", tag + dna, len(tag)); inc("tagged_full")
    for i in range(33):
        tag = TAG_PROK_2
        dna = seqs[prok_idx[base_prok + 33 + i]][:8192 - len(tag)]
        emit(out, "tagged_full", prok_idx[base_prok+33+i], "prok", tag + dna, len(tag)); inc("tagged_full")
    for i in range(34):
        tag = TAG_EUK
        dna = seqs[euk_idx[base_euk + i]][:8192 - len(tag)]
        emit(out, "tagged_full", euk_idx[base_euk+i], "euk", tag + dna, len(tag)); inc("tagged_full")
    base_prok += 66; base_euk += 34

    # ---- tagged_partial (50): 25 prok + 25 euk ----
    for i in range(25):
        tag = TAG_PARTIAL_PROK
        dna = seqs[prok_idx[base_prok + i]][:8192 - len(tag)]
        emit(out, "tagged_partial", prok_idx[base_prok+i], "prok", tag + dna, len(tag)); inc("tagged_partial")
    for i in range(25):
        tag = TAG_PARTIAL_EUK
        dna = seqs[euk_idx[base_euk + i]][:8192 - len(tag)]
        emit(out, "tagged_partial", euk_idx[base_euk+i], "euk", tag + dna, len(tag)); inc("tagged_partial")
    base_prok += 25; base_euk += 25

    # ---- tagged_minimal (50): 25 prok + 25 euk ----
    for i in range(25):
        tag = TAG_MIN_PROK
        dna = seqs[prok_idx[base_prok + i]][:8192 - len(tag)]
        emit(out, "tagged_minimal", prok_idx[base_prok+i], "prok", tag + dna, len(tag)); inc("tagged_minimal")
    for i in range(25):
        tag = TAG_MIN_EUK
        dna = seqs[euk_idx[base_euk + i]][:8192 - len(tag)]
        emit(out, "tagged_minimal", euk_idx[base_euk+i], "euk", tag + dna, len(tag)); inc("tagged_minimal")
    base_prok += 25; base_euk += 25

    # ---- length_4096 (100): 50 prok + 50 euk ----
    for i in range(50):
        emit(out, "length_4096", prok_idx[base_prok + i], "prok", seqs[prok_idx[base_prok+i]][:4096], 0); inc("length_4096")
        emit(out, "length_4096", euk_idx[base_euk + i],  "euk",  seqs[euk_idx[base_euk+i]][:4096],  0); inc("length_4096")
    base_prok += 50; base_euk += 50

    # ---- length_6144 (100): 50 prok + 50 euk ----
    for i in range(50):
        emit(out, "length_6144", prok_idx[base_prok + i], "prok", seqs[prok_idx[base_prok+i]][:6144], 0); inc("length_6144")
        emit(out, "length_6144", euk_idx[base_euk + i],  "euk",  seqs[euk_idx[base_euk+i]][:6144],  0); inc("length_6144")


print("\nemitted per condition:")
for k, v in count.items():
    print(f"  {k}: {v}")
print(f"\ntotal records: {sum(count.values())}")
print(f"out -> {OUT_FASTA}")
