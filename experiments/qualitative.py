"""Qualitative anomaly maps: image, ground truth, frozen memory head, RO-MVFA (fused score).

Examples are chosen by a fixed rule, not by inspection: among abnormal test images, sort by lesion
area and take the two images closest to the median area (ties broken by file order).
  python experiments/qualitative.py --backbone biomedclip --k 4 --set 1 --out paper_v3/figures/qualitative.pdf
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import fsad
from fsad import (DEV, EVAL, FSAD, Backbone, augment, cache_paths, knn_dist, load_img, load_mask, memory_bank,
                  norm, seed_all, support_set, train, upsample)


def pick(meta, masks, n=2):
    lab = np.array(meta["labels"])
    idx = np.where(lab == 1)[0]
    area = masks[idx].reshape(len(idx), -1).sum(1)
    order = idx[np.argsort(np.abs(area - np.median(area)), kind="stable")]
    return list(order[:n])


def maps_for(model, tok, mem_seg, frozen_bank_):
    t = tok.to(DEV).float()
    out = {}
    with torch.no_grad():
        fm = sum(upsample(knn_dist(t[:, l, 1:], frozen_bank_[l])) for l in range(t.shape[1]))
        xs = model.adapt(t, "seg")
        txt = sum(upsample(model.text_logits(x[:, 1:]).softmax(-1)[..., 1]) for x in xs)
        mem = sum(upsample(knn_dist(x[:, 1:], mem_seg[l])) for l, x in enumerate(xs))
    out["frozen"] = fm.cpu().numpy()
    out["text"], out["mem"] = txt.cpu().numpy(), mem.cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="biomedclip")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--set", type=int, default=1)
    ap.add_argument("--out", default=str(fsad.ROOT.parent / "paper_v3" / "figures" / "qualitative.pdf"))
    a = ap.parse_args()
    bb = Backbone(a.backbone)
    rows_fig = []
    for d in fsad.SEG_DATASETS:
        tok_p, meta_p, mask_p = cache_paths(a.backbone, d)
        meta = json.loads(meta_p.read_text())
        test_tok = np.load(tok_p, mmap_mode="r")
        masks = np.load(mask_p)
        bidx = [meta["blocks"].index(b) for b in (4, 8, 12)]
        sel = pick(meta, masks)
        # train RO-MVFA (MVFA objective, read-only adapters) on one support set
        seed_all(a.set * 100 + a.k)
        sn, sa, _ = support_set(d, a.k, a.set)
        rows = sa + sn
        imgs = torch.stack([load_img(r[0]) for r in rows])
        msk = torch.stack([load_mask(r[2]) for r in rows])
        lab0 = torch.tensor([r[1] for r in rows])
        ai, am, src = augment(imgs, msk)
        lab = lab0[src]
        with torch.no_grad():
            sup = torch.cat([bb.tokens(norm(ai[i:i + 32].to(DEV)), [4, 8, 12]).half().cpu() for i in range(0, len(ai), 32)])
        model = FSAD(3, bb.norm, bb.proj, bb.text(d), mode="dual").to(DEV)
        train(model, sup, am, lab, src, True, 50, 1e-3, 8)
        model.eval()
        normal = sup[lab == 0]
        mem_seg = memory_bank(model, normal, "seg")
        fb = fsad.frozen_bank(normal)
        tok = torch.from_numpy(np.ascontiguousarray(test_tok[sel][:, bidx]))
        m = maps_for(model, tok, mem_seg, fb)
        for j, i in enumerate(sel):
            img = np.asarray(load_img(meta["paths"][i], EVAL).permute(1, 2, 0))
            mm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12)
            fused = 0.5 * mm(m["text"][j]) + 0.5 * mm(m["mem"][j])
            rows_fig.append((d, img, masks[i], mm(m["frozen"][j]), fused))
        del model
    names = {"Brain": "Brain MRI", "Liver": "Liver CT", "Retina_RESC": "Retinal OCT (RESC)"}
    fig, ax = plt.subplots(len(rows_fig), 4, figsize=(6.6, 1.7 * len(rows_fig)))
    titles = ["Image", "Ground truth", "Frozen memory", "RO-MVFA"]
    for r, (d, img, gt, fr, fu) in enumerate(rows_fig):
        for c, im in enumerate([img, gt, fr, fu]):
            a_ = ax[r, c]
            if c == 0:
                a_.imshow(img)
                a_.set_ylabel(names[d], fontsize=7)
            elif c == 1:
                a_.imshow(img); a_.contour(gt, levels=[0.5], colors="lime", linewidths=0.8)
            else:
                a_.imshow(img); a_.imshow(im, cmap="jet", alpha=0.45, vmin=0, vmax=1)
                a_.contour(gt, levels=[0.5], colors="white", linewidths=0.5)
            a_.set_xticks([]); a_.set_yticks([])
            if r == 0:
                a_.set_title(titles[c], fontsize=8)
    plt.tight_layout(pad=0.3)
    fsad.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(a.out, dpi=300)
    plt.savefig(a.out.replace(".pdf", ".png"), dpi=150)
    print("saved", a.out)


if __name__ == "__main__":
    main()
