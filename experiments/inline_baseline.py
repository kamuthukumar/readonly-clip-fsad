"""In-line (MVFA-placement) adapters on the same ViT-B/16, support sets, losses and scoring as RO-MVFA.

Only the adapter placement differs: here the adapter outputs are mixed back into the residual
stream after blocks 4, 8 and 12 (x <- 0.8 x + 0.1 A_seg(x) + 0.1 A_det(x), as in MVFA), so every
training step and every test image needs a full encoder pass. Used to measure what the read-only
placement of RO-MVFA costs in accuracy and saves in time.

  python experiments/inline_baseline.py --datasets Brain Liver --shots 4 --sets 0 1 2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import fsad
from fsad import (DEV, EVAL, SEG_DATASETS, Backbone, FSAD, GPUFault, augment, check, dice_loss, focal_loss,
                  hist_auc, knn_dist, list_split, load_img, load_mask, norm, seed_all, support_set, upsample)

BLOCKS = [4, 8, 12]


class Inline:
    """Runs the frozen encoder with adapters injected after the tapped blocks."""

    def __init__(self, bb: Backbone, model: FSAD):
        self.bb, self.model = bb, model

    def __call__(self, x):
        feats = {"seg": [], "det": []}

        def hook(module, inp, out, l):
            o = out[0] if isinstance(out, tuple) else out
            s = self.model.seg[l](o)
            d = self.model.det[l](o)
            feats["seg"].append(s)
            feats["det"].append(d)
            new = 0.8 * o + 0.1 * s + 0.1 * d
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

        hooks = [self.bb.blocks[b - 1].register_forward_hook(lambda m, i, o, l=l: hook(m, i, o, l))
                 for l, b in enumerate(BLOCKS)]
        try:
            self.bb.model.encode_image(x)
        finally:
            for h in hooks:
                h.remove()
        return feats


def run(args, bb, dataset, k, set_id, log):
    seg = dataset in SEG_DATASETS
    seed_all(set_id * 100 + k)
    sup_n, sup_a, overlap = support_set(dataset, k, set_id)
    rows = sup_a + sup_n
    imgs = torch.stack([load_img(r[0]) for r in rows])
    masks = torch.stack([load_mask(r[2], EVAL) for r in rows])
    a_img, a_mask, src = augment(imgs, masks)
    labels = torch.tensor([r[1] for r in rows])[src].float()
    model = FSAD(len(BLOCKS), bb.norm, bb.proj, bb.text(dataset), mode="dual").to(DEV)
    enc = Inline(bb, model)
    params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("mem_")]
    opt = torch.optim.Adam(params, lr=1e-3, betas=(0.5, 0.999))
    X, M, Y = norm(a_img).to(DEV), a_mask.to(DEV), labels.to(DEV)

    t0 = time.time()
    model.train()
    for ep in range(args.epochs):
        perm = torch.randperm(len(X), device=DEV)
        for s in range(0, len(X), 8):
            idx = perm[s:s + 8]
            f = enc(X[idx])
            loss = 0.0
            for x in f["det"]:
                prob = model.text_logits(x[:, 1:]).softmax(-1)[..., 1]
                loss = loss + F.binary_cross_entropy(prob.mean(1).clamp(1e-6, 1 - 1e-6), Y[idx])
            if seg:
                m = M[idx][:, 0]
                for x in f["seg"]:
                    lg = model.text_logits(x[:, 1:])
                    B = lg.shape[0]
                    p2 = F.interpolate(lg.permute(0, 2, 1).reshape(B, 2, 14, 14), size=EVAL,
                                       mode="bilinear", align_corners=True).softmax(1)
                    loss = loss + focal_loss(p2, m) + dice_loss(p2[:, 1], m)
            if not torch.isfinite(loss):
                raise GPUFault("loss not finite")
            opt.zero_grad()
            loss.backward()
            opt.step()
    t_train = time.time() - t0
    model.eval()

    with torch.no_grad():
        normal = X[Y == 0]
        banks = {"seg": [[] for _ in BLOCKS], "det": [[] for _ in BLOCKS]}
        for s in range(0, len(normal), 16):
            f = enc(normal[s:s + 16])
            for h in banks:
                for l, x in enumerate(f[h]):
                    banks[h][l].append(F.normalize(x[:, 1:].reshape(-1, 768), dim=-1).half())
        banks = {h: [torch.cat(v) for v in banks[h]] for h in banks}

        test = list_split(dataset, "test")
        y = np.array([r[1] for r in test])
        n = len(test)
        det_t, det_m = torch.zeros(n), torch.zeros(n)
        seg_t = torch.zeros((n, EVAL, EVAL), dtype=torch.float16) if seg else None
        seg_m = torch.zeros((n, EVAL, EVAL), dtype=torch.float16) if seg else None
        gt = torch.zeros((n, EVAL, EVAL), dtype=torch.bool) if seg else None
        t1 = time.time()
        for s in range(0, n, args.eval_batch):
            chunk = test[s:s + args.eval_batch]
            x = norm(torch.stack([load_img(r[0]) for r in chunk]).to(DEV))
            f = enc(x)
            dt = sum(model.text_logits(z[:, 1:]).softmax(-1)[..., 1].mean(1) for z in f["det"])
            dm = sum(knn_dist(z[:, 1:], banks["det"][l]).mean(1) for l, z in enumerate(f["det"]))
            det_t[s:s + len(chunk)] = check(dt, "text scores").cpu()
            det_m[s:s + len(chunk)] = check(dm, "memory scores").cpu()
            if seg:
                st_ = sum(upsample(model.text_logits(z[:, 1:]).softmax(-1)[..., 1]) for z in f["seg"])
                sm = sum(upsample(knn_dist(z[:, 1:], banks["seg"][l])) for l, z in enumerate(f["seg"]))
                seg_t[s:s + len(chunk)] = st_.half().cpu()
                seg_m[s:s + len(chunk)] = sm.half().cpu()
                for j, r in enumerate(chunk):
                    gt[s + j] = load_mask(r[2])[0] > 0.5
        t_eval = time.time() - t1

    mm = lambda a: (a.float() - a.float().min()) / (a.float().max() - a.float().min() + 1e-12)
    res = {"dataset": dataset, "k": k, "set": set_id, "backbone": args.backbone, "config": "inline_dual_L4-8-12_mvfa",
           "trainable_params": sum(p.numel() for p in params), "sec_train": round(t_train, 1),
           "sec_eval": round(t_eval, 1), "ms_per_test_image_total": round(1000 * t_eval / n, 2),
           "support_in_test": overlap, "variants": {}}
    v = res["variants"]
    v["image_det_testminmax"] = float(roc_auc_score(y, (0.5 * mm(det_t) + 0.5 * mm(det_m)).numpy()))
    v["image_det_memory"] = float(roc_auc_score(y, det_m.numpy()))
    if seg:
        fused = 0.5 * mm(seg_t) + 0.5 * mm(seg_m)
        v["pixel_testminmax"] = hist_auc(fused.flatten(), gt.flatten())
        v["image_max_testminmax"] = float(roc_auc_score(y, fused.flatten(1).max(1).values.numpy()))
        v["pixel_memory"] = hist_auc(seg_m.float().flatten(), gt.flatten())
        v["image_max_memory"] = float(roc_auc_score(y, seg_m.float().flatten(1).max(1).values.numpy()))
    log.write(json.dumps(res) + "\n")
    log.flush()
    key = "image_max_testminmax" if seg else "image_det_testminmax"
    print(f"  inline {dataset:15s} k={k} set={set_id} image {100 * v[key]:.2f}"
          + (f" pixel {100 * v['pixel_testminmax']:.2f}" if seg else "")
          + f" (train {t_train:.0f}s, eval {t_eval:.0f}s)", flush=True)
    del model, enc, X, M, banks
    if DEV.type == "mps":
        torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="openai", choices=list(fsad.BACKBONES))
    ap.add_argument("--datasets", nargs="+", default=fsad.ALL_DATASETS)
    ap.add_argument("--shots", type=int, nargs="+", default=[4])
    ap.add_argument("--sets", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--eval_batch", type=int, default=32)
    ap.add_argument("--out", default=str(fsad.ROOT.parent / "results_fsad"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log_p = out / f"{a.backbone}__inline_dual_L4-8-12_mvfa.jsonl"
    done = set()
    if log_p.exists():
        done = {(r["dataset"], r["k"], r["set"]) for r in map(json.loads, log_p.read_text().splitlines())}
    bb = Backbone(a.backbone)
    with log_p.open("a") as log:
        for d in a.datasets:
            for k in a.shots:
                for s in a.sets:
                    if (d, k, s) in done:
                        continue
                    for attempt in range(3):
                        try:
                            run(a, bb, d, k, s, log)
                            break
                        except GPUFault as e:
                            print(f"  GPU fault ({e}); retry", flush=True)
                            torch.mps.empty_cache() if DEV.type == "mps" else None
                            time.sleep(30)


if __name__ == "__main__":
    main()
