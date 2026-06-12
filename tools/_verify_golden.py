"""Verify a candidate metrics.npz is identical to the golden (Phase-1 dev aid)."""
import sys
import numpy as np

golden = np.load(sys.argv[1], allow_pickle=True)
cand = np.load(sys.argv[2], allow_pickle=True)

gk, ck = set(golden.files), set(cand.files)
if gk != ck:
    print("KEY MISMATCH")
    print("  only in golden:", sorted(gk - ck))
    print("  only in cand  :", sorted(ck - gk))
    sys.exit(1)

mism = []
for k in golden.files:
    a, b = golden[k], cand[k]
    if a.shape != b.shape:
        mism.append((k, f"shape {a.shape} vs {b.shape}"))
        continue
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        af, bf = a.astype(np.float64), b.astype(np.float64)
        if np.array_equal(np.nan_to_num(af), np.nan_to_num(bf)):
            continue
        d = float(np.nanmax(np.abs(af - bf)))
        rel = d / (float(np.nanmax(np.abs(af))) + 1e-30)
        mism.append((k, f"maxabs={d:.3e} maxrel={rel:.3e}"))
    else:
        if not np.array_equal(a, b):
            mism.append((k, "value diff"))

if not mism:
    print(f"IDENTICAL: all {len(golden.files)} arrays match the golden exactly.")
    sys.exit(0)
print(f"DIFFERENCES in {len(mism)}/{len(golden.files)} arrays:")
for k, msg in mism[:40]:
    print(f"  {k}: {msg}")
sys.exit(2)
