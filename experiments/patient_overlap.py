"""Volume overlap between the support sets and the test split, read from the file names
(<volume>_<slice>.png). Reports, per dataset, how many abnormal test images share a volume with
the abnormal support images, because the lesion memory stores patches from those images."""
import sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import fsad

vol = lambda p: Path(p).stem.rsplit("_", 1)[0]
for d in ["Brain", "Liver", "Retina_RESC"]:
    test = fsad.list_split(d, "test")
    valid = fsad.list_split(d, "valid")
    tv_a = collections.Counter(vol(r[0]) for r in test if r[1] == 1)
    tv_n = collections.Counter(vol(r[0]) for r in test if r[1] == 0)
    va = {vol(r[0]) for r in valid if r[1] == 1}
    print(f"{d}: abnormal valid volumes {len(va)}, also in abnormal test {len(va & set(tv_a))}, "
          f"test volumes abnormal {len(tv_a)}, normal {len(tv_n)}, test abnormal images {sum(tv_a.values())}")
    for k in [2, 4, 8, 16]:
        out = []
        for s in [0, 1, 2]:
            _, sup_a, _ = fsad.support_set(d, k, s)
            sv = {vol(r[0]) for r in sup_a}
            hit = sum(c for v, c in tv_a.items() if v in sv)
            out.append(f"s{s}: {len(sv)} vols, {100 * hit / sum(tv_a.values()):.1f}%")
        print(f"  k={k:<2d} abnormal test images sharing a volume with lesion support: " + " | ".join(out))
