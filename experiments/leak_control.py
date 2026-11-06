"""Leakage control. On identical test slices (one half of the patients), compare support images
drawn from those same patients ("same") with support images drawn from the other patients
("other"), so that the paired difference cannot come from a change of test population.
Finished pairs are skipped. Results: results_leak/<dataset>_<mode>.jsonl"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import fsad

vol = lambda p: Path(p).stem.rsplit("_", 1)[0]
OUT = fsad.ROOT.parent / "results_leak"


def make_args():
    return argparse.Namespace(backbone="biomedclip", adapter="dual", blocks=[4, 8, 12], fusion="testminmax",
                              heads="both", prompts="mvfa", epochs=50, lr=1e-3, batch=8, gamma0=0.01,
                              w_text_loss=1.0, w_mem_loss=0.0, w_text=0.5, image_score="max", dual_memory=False,
                              det_pool="mean", eval_batch=32, eval_every=0, dev=False, extra_scores=True,
                              xs_lite=True, ensemble=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Liver", "Retina_RESC"])
    ap.add_argument("--shots", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--partitions", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    args = make_args()
    bb = fsad.Backbone(args.backbone)
    original_support = fsad.support_set
    for d in a.datasets:
        tok_p, meta_p, mask_p = fsad.cache_paths(args.backbone, d, "test")
        meta = json.loads(meta_p.read_text())
        tok = np.load(tok_p, mmap_mode="r")
        masks = np.load(mask_p)
        bidx = [meta["blocks"].index(b) for b in args.blocks]
        text = bb.text(d, args.prompts)
        valid = fsad.list_split(d, "valid")
        test_h = fsad.test_hashes(d)
        pool = [r for r in valid if fsad.image_hash(r[0]) not in test_h]
        ids = sorted({vol(r[0]) for r in valid} | {vol(p) for p in meta["paths"]}, key=lambda s: int(s))
        tvol = np.array([vol(p) for p in meta["paths"]])
        for p in a.partitions:
            perm = ids[:]
            random.Random(5000 + p).shuffle(perm)
            halves = [set(perm[:len(perm) // 2]), set(perm[len(perm) // 2:])]
            for f in (0, 1):
                T = halves[f]
                idx = np.nonzero(np.isin(tvol, sorted(T)))[0]
                sub_tok, sub_masks = None, masks[idx]
                sub_meta = {"n": len(idx), "labels": [meta["labels"][i] for i in idx],
                            "paths": [meta["paths"][i] for i in idx]}
                for k in a.shots:
                    set_id = 10 * (p + 1) + f
                    for mode, allowed in (("same", T), ("other", halves[1 - f])):
                        log_p = OUT / f"{d}_{mode}.jsonl"
                        done = {(r["k"], r["set"]) for r in map(json.loads, log_p.read_text().splitlines())} \
                            if log_p.exists() else set()
                        if (k, set_id) in done:
                            continue
                        n_pool = [r for r in pool if r[1] == 0 and vol(r[0]) in allowed]
                        a_pool = [r for r in pool if r[1] == 1 and vol(r[0]) in allowed]
                        if len(n_pool) < k or len(a_pool) < k:
                            print(f"  SKIP {d} k={k} set={set_id} {mode}: {len(n_pool)}/{len(a_pool)} support candidates", flush=True)
                            continue
                        rng = random.Random(7000 + 100 * set_id + k)
                        sup = (rng.sample(n_pool, k), rng.sample(a_pool, k), 0)
                        fsad.support_set = lambda *_: sup
                        if sub_tok is None:
                            sub_tok = np.array(tok[idx][:, bidx])
                        with log_p.open("a") as log:
                            print(f"[{mode}]", end="", flush=True)
                            fsad.run_one(bb, args, d, k, set_id, sub_tok, sub_meta, sub_masks, text, [0, 1, 2], log)
                        fsad.support_set = original_support
                del sub_tok
    print("LEAK CONTROL DONE", flush=True)


if __name__ == "__main__":
    main()
