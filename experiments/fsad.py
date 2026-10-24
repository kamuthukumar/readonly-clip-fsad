"""Few-shot medical anomaly detection and localisation on BMAD with read-only CLIP adapters.

Protocol (identical to MVFA, Huang et al. CVPR 2024, except where stated):
  * target dataset only: k normal + k abnormal support images from the BMAD *valid*
    split, augmented 11x (4 rotations, 4 translations, 2 flips) as in MVFA;
  * evaluation on the full official BMAD *test* split; image AUC for all six
    datasets, pixel AUC over all test pixels for Brain, Liver and Retina_RESC;
  * support set "0" is MVFA's published list (experiments/fewshot_seed); sets 1 and 2
    are drawn at random from the valid split with a fixed seed;
  * unlike MVFA we never select epochs on the test set: every run trains for a fixed
    number of epochs. The paper's model keeps MVFA's min-max fusion over the test set
    (``--fusion testminmax``) for comparability, and every run also records the
    memory-only score, which uses no test-set statistics (``variants["*_memory"]``).

The backbone is frozen and the adapters only *read* intermediate tokens, so test
features are extracted once per backbone and cached; every k, seed and ablation reuses
them. Commands:

  python experiments/fsad.py cache --backbone openai --datasets Brain Liver Retina_RESC
  python experiments/fsad.py run --backbone openai --datasets Brain --shots 2 4 8 16 --sets 0 1 2
  python experiments/fsad.py table --out results_fsad
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("BMAD_ROOT", "~/data/BMAD")).expanduser()
CACHE = Path(os.environ.get("FSAD_CACHE", "~/data/fsad_cache")).expanduser()

SEG_DATASETS = ["Brain", "Liver", "Retina_RESC"]
CLS_DATASETS = ["Retina_OCT2017", "Chest", "Histopathology"]
ALL_DATASETS = SEG_DATASETS + CLS_DATASETS
# MVFA's class names for the prompt ensemble (prompt.py, REAL_NAME)
REAL_NAME = {"Brain": "Brain", "Liver": "Liver", "Retina_RESC": "retinal OCT", "Chest": "Chest X-ray film",
             "Retina_OCT2017": "retinal OCT", "Histopathology": "histopathological image"}
# Clinically phrased names (used with --prompts medical)
MED_NAME = {"Brain": "brain MRI", "Liver": "abdominal CT of the liver", "Retina_RESC": "retinal OCT B-scan",
            "Chest": "chest radiograph", "Retina_OCT2017": "retinal OCT B-scan",
            "Histopathology": "histopathology patch of a lymph node"}
MED_NORMAL = ["{}", "normal {}", "healthy {}", "{} with no abnormality", "{} without lesion", "unremarkable {}"]
MED_ABNORMAL = ["abnormal {}", "{} with a lesion", "{} with a tumor", "{} showing pathology", "diseased {}",
                "{} with an abnormal finding"]
MVFA_NORMAL = ["{}", "flawless {}", "perfect {}", "unblemished {}", "{} without flaw", "{} without defect",
               "{} without damage"]
MVFA_ABNORMAL = ["damaged {}", "broken {}", "{} with flaw", "{} with defect", "{} with damage"]
TEMPLATES = ["a bad photo of a {}.", "a low resolution photo of the {}.", "a bad photo of the {}.",
             "a cropped photo of the {}.", "a bright photo of a {}.", "a dark photo of the {}.", "a photo of my {}.",
             "a photo of the cool {}.", "a close-up photo of a {}.", "a black and white photo of the {}.",
             "a bright photo of the {}.", "a cropped photo of a {}.", "a jpeg corrupted photo of a {}.",
             "a blurry photo of the {}.", "a photo of the {}.", "a good photo of the {}.", "a photo of one {}.",
             "a close-up photo of the {}.", "a photo of a {}.", "a low resolution photo of a {}.",
             "a photo of a large {}.", "a blurry photo of a {}.", "a jpeg corrupted photo of the {}.",
             "a good photo of a {}.", "a photo of the small {}.", "a photo of the large {}.",
             "a black and white photo of a {}.", "a dark photo of a {}.", "a photo of a cool {}.",
             "a photo of a small {}.", "there is a {} in the scene.", "there is the {} in the scene.",
             "this is a {} in the scene.", "this is the {} in the scene.", "this is one {} in the scene."]

BACKBONES = {
    "openai": ("ViT-B-16-quickgelu", "openai"),
    "biomedclip": ("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", None),
}
CACHE_BLOCKS = [3, 4, 6, 8, 9, 12]  # 1-indexed transformer blocks stored in the cache
IMG = 224    # network input
EVAL = 240   # masks and maps are compared at MVFA's resolution
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEV = device()


class GPUFault(RuntimeError):
    """Raised when the GPU returns zeros or NaN (on Apple MPS this is what an out-of-memory
    command buffer looks like); the run is retried and never written to the results."""


def check(x, what):
    if not torch.isfinite(x).all() or x.abs().max() == 0:
        raise GPUFault(f"{what}: non-finite or all-zero values")
    return x


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class Backbone:
    """Frozen open_clip model with read-only taps on intermediate blocks."""

    def __init__(self, name):
        import open_clip
        arch, pre = BACKBONES[name]
        kw = {"pretrained": pre} if pre else {}
        self.model = open_clip.create_model_and_transforms(arch, **kw)[0].to(DEV).eval()
        self.tokenizer = open_clip.get_tokenizer(arch)
        v = self.model.visual
        if hasattr(v, "trunk"):  # timm ViT (BiomedCLIP)
            self.blocks, self.norm = v.trunk.blocks, v.trunk.norm
            self.proj = v.head.proj.weight.T.detach()  # 768 x 512
        else:  # open_clip ViT (OpenAI)
            self.blocks, self.norm = v.transformer.resblocks, v.ln_post
            self.proj = v.proj.detach()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def tokens(self, x, blocks):
        """x: [B,3,224,224] normalised -> [B, len(blocks), 197, 768]."""
        out = {}
        hooks = [self.blocks[b - 1].register_forward_hook(
            lambda m, i, o, b=b: out.__setitem__(b, o[0] if isinstance(o, tuple) else o)) for b in blocks]
        try:
            self.model.encode_image(x)
        finally:
            for h in hooks:
                h.remove()
        return torch.stack([out[b] for b in blocks], 1)

    @torch.no_grad()
    def text(self, dataset, style="mvfa"):
        name = (REAL_NAME if style == "mvfa" else MED_NAME)[dataset]
        states = (MVFA_NORMAL, MVFA_ABNORMAL) if style == "mvfa" else (MED_NORMAL, MED_ABNORMAL)
        feats = []
        for state in states:
            sents = [t.format(s.format(name)) for s in state for t in TEMPLATES]
            e = self.model.encode_text(self.tokenizer(sents).to(DEV)).float()
            e = F.normalize(e, dim=-1).mean(0)
            feats.append(F.normalize(e, dim=0))
        return torch.stack(feats, 1)  # 512 x 2 (normal, abnormal)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def split_dir(dataset, split, label):
    base = DATA / dataset / split
    for cand in ([label] if label == "good" else ["Ungood", "ungood"]):
        if (base / cand).exists():
            return base / cand
    raise FileNotFoundError(f"{base}/{label}")


def list_split(dataset, split):
    """[(image_path, label, mask_path or None)], normals first, sorted like MVFA."""
    rows = []
    for label, lab in (("good", 0), ("Ungood", 1)):
        d = split_dir(dataset, split, label)
        img_dir = d / "img" if (d / "img").exists() else d
        for p in sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpeg")) + sorted(img_dir.glob("*.jpg")):
            m = d / "label" / p.name
            rows.append((str(p), lab, str(m) if (dataset in SEG_DATASETS and lab == 1 and m.exists()) else None))
    return rows


def load_img(path, size=IMG):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)


def load_mask(path, size=EVAL):
    if path is None:
        return torch.zeros(1, size, size)
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return torch.from_numpy((np.asarray(m) > 127).astype(np.float32))[None]


def norm(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def image_hash(path):
    import hashlib
    return hashlib.md5(np.asarray(Image.open(path).convert("L").resize((64, 64))).tobytes()).hexdigest()


def test_hashes(dataset):
    """Pixel hashes of the test split (cached). BMAD's Brain valid split contains
    images that are pixel-identical to test images; these must not enter random support sets."""
    p = CACHE / "hashes" / f"{dataset}_test.json"
    if p.exists():
        return set(json.loads(p.read_text()))
    hs = sorted({image_hash(r[0]) for r in list_split(dataset, "test")})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(hs))
    return set(hs)


def train_normals(dataset):
    d = DATA / dataset / "train" / "good"
    if not d.exists():
        return []
    img_dir = d / "img" if (d / "img").exists() else d
    return [(str(p), 0, None) for p in sorted(img_dir.glob("*.png"))]


def support_set(dataset, k, set_id):
    """k normal + k abnormal support images. Set 0 is MVFA's published list; sets >= 1 are
    random draws from the valid split (normals topped up from the normal-only train split
    when valid has fewer than k, which only happens for Retina_OCT2017 at k=16).
    Returns the two lists and the number of support images pixel-identical to a test image."""
    rows = list_split(dataset, "valid")
    normals = [r for r in rows if r[1] == 0]
    abnormals = [r for r in rows if r[1] == 1]
    extra = train_normals(dataset)
    test_h = test_hashes(dataset)
    if set_id == 0:
        lines = dict(line.split(":", 1) for line in (ROOT / "fewshot_seed" / dataset / f"{k}-shot.txt").read_text().split("\n") if ":" in line)
        # match on the stem: MVFA lists OCT2017 files as .jpeg, the BMAD mirror stores .png
        def pick(pool, key):
            out = []
            for name in lines[key].split():
                hit = [r for r in pool if Path(r[0]).stem == Path(name).stem]
                out.append(hit[0] if hit else None)
            return out
        sup_n = [r for r in pick(normals + extra, "n-0") if r]
        sup_a = [r for r in pick(abnormals, "a-0") if r]
    else:
        rng = random.Random(1000 * set_id + k)
        clean = lambda pool: [r for r in pool if image_hash(r[0]) not in test_h]
        pool_n = clean(normals)
        if len(pool_n) < k:
            pool_n = pool_n + rng.sample(clean(extra), k - len(pool_n))
        sup_n, sup_a = rng.sample(pool_n, k), rng.sample(clean(abnormals), k)
    assert len(sup_n) == k and len(sup_a) == k, f"{dataset} k={k} set={set_id}: {len(sup_n)}/{len(sup_a)}"
    overlap = sum(image_hash(r[0]) in test_h for r in sup_n + sup_a)
    return sup_n, sup_a, overlap


def _affine(x, theta, mode):
    grid = F.affine_grid(theta.expand(x.shape[0], 2, 3), x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode=mode, padding_mode="reflection", align_corners=False)


def augment(imgs, masks):
    """MVFA augmentation: original, 4 rotations, 4 translations, h-flip, v-flip (11x)."""
    outs_i, outs_m = [imgs], [masks]
    for a in (-math.pi / 8, -math.pi / 16, math.pi / 16, math.pi / 8):
        th = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0]], dtype=torch.float32)
        outs_i.append(_affine(imgs, th, "bilinear"))
        outs_m.append((_affine(masks, th, "nearest") > 0.5).float())
    for a, b in ((0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1)):
        th = torch.tensor([[1, 0, a], [0, 1, b]], dtype=torch.float32)
        outs_i.append(_affine(imgs, th, "bilinear"))
        outs_m.append((_affine(masks, th, "nearest") > 0.5).float())
    outs_i += [imgs.flip(-1), imgs.flip(-2)]
    outs_m += [masks.flip(-1), masks.flip(-2)]
    src = torch.arange(imgs.shape[0]).repeat(len(outs_i))  # which original image each copy came from
    return torch.cat(outs_i), torch.cat(outs_m), src


# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------


def cache_paths(backbone, dataset, split="test"):
    d = CACHE / backbone
    return d / f"{dataset}_{split}_tokens.npy", d / f"{dataset}_{split}_meta.json", d / f"{dataset}_{split}_masks.npy"


def build_cache(backbone_name, dataset, blocks, batch=32, split="test", bb=None):
    tok_p, meta_p, mask_p = cache_paths(backbone_name, dataset, split)
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta.get("complete") and meta["blocks"] == blocks:
            print(f"cache ok: {tok_p}")
            return
    tok_p.parent.mkdir(parents=True, exist_ok=True)
    rows = list_split(dataset, split)
    bb = bb or Backbone(backbone_name)
    n = len(rows)
    tokens = np.lib.format.open_memmap(tok_p, mode="w+", dtype=np.float16, shape=(n, len(blocks), 197, 768))
    masks = np.zeros((n, EVAL, EVAL), dtype=np.uint8) if dataset in SEG_DATASETS else None
    t0 = time.time()
    for s in range(0, n, batch):
        chunk = rows[s:s + batch]
        x = norm(torch.stack([load_img(r[0]) for r in chunk]).to(DEV))
        tokens[s:s + len(chunk)] = bb.tokens(x, blocks).half().cpu().numpy()
        if masks is not None:
            for j, r in enumerate(chunk):
                masks[s + j] = load_mask(r[2])[0].numpy().astype(np.uint8)
        if (s // batch) % 20 == 0:
            print(f"  {dataset}: {s + len(chunk)}/{n}  {(s + len(chunk)) / (time.time() - t0):.1f} img/s", flush=True)
    tokens.flush()
    if masks is not None:
        np.save(mask_p, masks)
    meta_p.write_text(json.dumps({"blocks": blocks, "n": n, "labels": [r[1] for r in rows],
                                  "paths": [r[0] for r in rows], "complete": True}))
    print(f"cached {n} images of {dataset} in {time.time() - t0:.0f}s -> {tok_p}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class GatedAdapter(nn.Module):
    """x + gamma * (W_up GELU(W_down x) - x): residual bottleneck with a learnable gate."""

    def __init__(self, dim=768, ratio=0.25, gamma0=0.01):
        super().__init__()
        self.down = nn.Linear(dim, int(dim * ratio))
        self.up = nn.Linear(int(dim * ratio), dim)
        self.gamma = nn.Parameter(torch.tensor(float(gamma0)))

    def forward(self, x):
        return (1 - self.gamma) * x + self.gamma * self.up(F.gelu(self.down(x)))


class FSAD(nn.Module):
    """Adapters on cached tokens. mode: shared (one set for both heads) | dual (MVFA-style) | none."""

    def __init__(self, n_levels, norm_layer, proj, text, mode="shared", gamma0=0.01, ratio=0.25):
        super().__init__()
        self.mode = mode
        make = lambda: nn.ModuleList([GatedAdapter(768, ratio, gamma0) for _ in range(n_levels)])
        self.seg = make() if mode != "none" else None
        self.det = (make() if mode == "dual" else None)
        self.norm = norm_layer
        # per-level scale and bias that turn a memory distance into an anomaly logit (memory loss only)
        self.mem_scale = nn.Parameter(torch.full((n_levels,), 20.0))
        self.mem_bias = nn.Parameter(torch.full((n_levels,), -2.0))
        self.register_buffer("proj", proj.float())
        self.register_buffer("text", text.float())

    def adapt(self, tok, head):
        """tok [B, L, N, D] -> list of adapted [B, N, D] per level."""
        bank = self.det if (head == "det" and self.mode == "dual") else self.seg
        return [tok[:, l] if bank is None else bank[l](tok[:, l]) for l in range(tok.shape[1])]

    def text_logits(self, x):
        v = F.normalize(self.norm(x) @ self.proj, dim=-1)
        return 100.0 * v @ self.text  # [B, N, 2]


def focal_loss(prob2, target, gamma=2.0, eps=1e-6):
    p = prob2[:, 1].clamp(eps, 1 - eps)
    pt = torch.where(target > 0.5, p, 1 - p)
    return (-(1 - pt) ** gamma * pt.log()).mean()


def dice_loss(p, target, eps=1.0):
    inter = (p * target).sum((-2, -1))
    return (1 - (2 * inter + eps) / (p.sum((-2, -1)) + target.sum((-2, -1)) + eps)).mean()


def upsample(patch_scores, size=EVAL):
    """[B, 196] -> [B, size, size]"""
    B, N = patch_scores.shape
    g = int(math.sqrt(N))
    return F.interpolate(patch_scores.view(B, 1, g, g), size=size, mode="bilinear", align_corners=True)[:, 0]


def patch_labels(masks, grid=14):
    """[B,1,H,W] pixel masks -> [B, grid*grid] patch labels (patch counts as abnormal if >25% lesion)."""
    return (F.adaptive_avg_pool2d(masks, grid).flatten(1) > 0.25).float()


def memory_distance(q, bank):
    """q [B,N,D], bank [M,D] (both L2-normalised) -> [B,N] distance to the nearest bank patch."""
    return 1 - (q @ bank.T).max(-1).values


def train(model, tok, masks, labels, src, seg, epochs, lr, batch, w_text=1.0, w_mem=0.0, n_mem=4096,
          on_epoch=None):
    """Text-alignment losses as in MVFA (w_text) plus, optionally, a memory-consistency loss
    (w_mem): adapted normal patches should lie close to the normal support memory and
    lesion patches far from it, the quantity the memory head scores at test time. The memory
    is rebuilt from the current adapters at every step, excluding patches from the same
    source image, so a normal image is never matched to itself."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params or epochs == 0:
        return
    opt = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))
    n = tok.shape[0]
    # keep the (small) support set on the device for the whole run
    plabels = patch_labels(masks).to(DEV) if seg else None  # pooled on CPU: MPS lacks non-divisible adaptive pooling
    tok, labels, src = tok.to(DEV), labels.to(DEV), src.to(DEV)
    masks = masks.to(DEV)
    normal_idx = torch.nonzero(labels == 0).flatten()
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            t = tok[idx].float()
            y = labels[idx].float()
            loss = 0.0
            if w_text:
                for x in model.adapt(t, "det"):
                    prob = model.text_logits(x[:, 1:]).softmax(-1)[..., 1]
                    loss = loss + w_text * F.binary_cross_entropy(prob.mean(1).clamp(1e-6, 1 - 1e-6), y)
                if seg:
                    m = masks[idx][:, 0]
                    for x in model.adapt(t, "seg"):
                        logit = model.text_logits(x[:, 1:])  # [B, 196, 2]
                        B = logit.shape[0]
                        lg = F.interpolate(logit.permute(0, 2, 1).reshape(B, 2, 14, 14), size=EVAL,
                                           mode="bilinear", align_corners=True)
                        prob2 = lg.softmax(1)
                        loss = loss + w_text * (focal_loss(prob2, m) + dice_loss(prob2[:, 1], m))
            if w_mem:
                # memory: a random subset of normal support images; each query image ignores
                # patches that come from its own source image (any augmentation of it)
                keep = normal_idx[torch.randperm(len(normal_idx), device=DEV)[: max(2, n_mem // 196 + 1)]]
                mem_tok = tok[keep].float()
                own = src[idx].view(-1, 1) == src[keep].view(1, -1)                     # [B, Mimg]
                own = own.repeat_interleave(196, dim=1).unsqueeze(1)                   # [B, 1, Mimg*196]
                head = "seg" if seg else "det"
                for l, (x, xm) in enumerate(zip(model.adapt(t, head), model.adapt(mem_tok, head))):
                    q = F.normalize(x[:, 1:], dim=-1)
                    bank = F.normalize(xm[:, 1:], dim=-1).reshape(-1, 768)
                    sim = (q @ bank.T).masked_fill(own, -2.0)
                    d = 1 - sim.max(-1).values                          # [B, 196]
                    logit = model.mem_scale[l] * d + model.mem_bias[l]
                    if seg:
                        loss = loss + w_mem * F.binary_cross_entropy_with_logits(logit, plabels[idx])
                    img_logit = model.mem_scale[l] * d.max(1).values + model.mem_bias[l]
                    loss = loss + w_mem * F.binary_cross_entropy_with_logits(img_logit, y)
            if not torch.isfinite(loss):
                raise GPUFault("training loss is not finite")
            opt.zero_grad()
            loss.backward()
            opt.step()
        if on_epoch is not None:
            on_epoch(ep + 1)


@torch.no_grad()
def memory_bank(model, tok_normal, head):
    return [F.normalize(x[:, 1:].reshape(-1, 768), dim=-1).half() for x in model.adapt(tok_normal.to(DEV).float(), head)]


@torch.no_grad()
def frozen_bank(tok_normal):
    """Memory of un-adapted CLIP patch tokens (one bank per tapped level)."""
    t = tok_normal.to(DEV).float()
    return [F.normalize(t[:, l, 1:].reshape(-1, 768), dim=-1).half() for l in range(t.shape[1])]


def knn_dist(p, bank, chunk=8):
    """p [B,196,768] -> [B,196] cosine distance to the nearest memory patch."""
    q = F.normalize(p, dim=-1).half()
    return torch.cat([(1 - (q[i:i + chunk].reshape(-1, 768) @ bank.T).max(1).values.float()).view(-1, 196)
                      for i in range(0, q.shape[0], chunk)])


@torch.no_grad()
def score_batch(model, t, mem_seg, mem_det, seg, chunk=8, det_pool="mean", mem_frozen=None):
    """Per-image text and memory scores and, for segmentation datasets, per-pixel maps.
    With mem_frozen, the memory score adds the distance measured in the frozen CLIP space
    to the distance in the adapted space (dual memory)."""
    t = t.to(DEV).float()
    out = {}
    det_text, det_mem = 0.0, 0.0
    for l, x in enumerate(model.adapt(t, "det")):
        p = x[:, 1:]
        det_text = det_text + model.text_logits(p).softmax(-1)[..., 1].mean(1)
        d = knn_dist(p, mem_det[l], chunk)
        if mem_frozen is not None:
            d = d + knn_dist(t[:, l, 1:], mem_frozen[l], chunk)
        det_mem = det_mem + (d.max(1).values if det_pool == "max" else d.mean(1))
    out["det_text"], out["det_mem"] = det_text, det_mem
    if seg:
        seg_text, seg_mem = 0.0, 0.0
        for l, x in enumerate(model.adapt(t, "seg")):
            p = x[:, 1:]
            seg_text = seg_text + upsample(model.text_logits(p).softmax(-1)[..., 1])
            d = knn_dist(p, mem_seg[l], chunk)
            if mem_frozen is not None:
                d = d + knn_dist(t[:, l, 1:], mem_frozen[l], chunk)
            seg_mem = seg_mem + upsample(d)
        out["seg_text"], out["seg_mem"] = seg_text, seg_mem
    return out


@torch.no_grad()
def support_stats(model, tok, src, labels, seg, chunk=8, det_pool="mean", dual_memory=False):
    """Mean/std of each head's scores on the normal support images (leave-one-source-image-out
    for the memory head), so that the text and memory scores can be put on one scale without
    looking at test data. With dual_memory the frozen-space distance is included, as at test time."""
    normal = labels == 0
    tn, sn = tok[normal].to(DEV).float(), src[normal].to(DEV)
    owner = sn.repeat_interleave(196)

    def loo_dist(q):  # q [n,196,D] normalised -> leave-one-source-out nearest-neighbour distance
        bank = q.reshape(-1, q.shape[-1])
        out = []
        for i in range(q.shape[0]):
            sim = q[i] @ bank.T
            sim[:, owner == sn[i]] = -2
            out.append(1 - sim.max(1).values)
        return torch.stack(out)

    stats = {}
    for head in (["det", "seg"] if seg else ["det"]):
        txt, mem = 0.0, 0.0
        for l, x in enumerate(model.adapt(tn, head)):
            p = x[:, 1:]
            pr = model.text_logits(p).softmax(-1)[..., 1]
            dl = loo_dist(F.normalize(p, dim=-1))
            if dual_memory:
                dl = dl + loo_dist(F.normalize(tn[:, l, 1:], dim=-1))
            if head == "det":
                txt = txt + pr.mean(1)
                mem = mem + (dl.max(1).values if det_pool == "max" else dl.mean(1))
            else:
                txt, mem = txt + upsample(pr), mem + upsample(dl)
        # floor the spread: a saturated head can be near-constant on the support normals
        spread = lambda v: max(v.std().item(), 1e-3 * abs(v.mean().item()), 1e-6)
        stats[head] = {"text": (txt.mean().item(), spread(txt)), "mem": (mem.mean().item(), spread(mem))}
    return stats


def hist_auc(scores, labels, bins=1 << 16, chunk=1 << 24):
    """ROC-AUC over very large pixel arrays via fine score bins, processed in chunks."""
    lo, hi = float(scores.min()), float(scores.max())
    pos = torch.zeros(bins, dtype=torch.float64)
    neg = torch.zeros(bins, dtype=torch.float64)
    for s in range(0, scores.numel(), chunk):
        sc, lb = scores[s:s + chunk].float(), labels[s:s + chunk]
        idx = ((sc - lo) / (hi - lo + 1e-12) * (bins - 1)).long().clamp_(0, bins - 1)
        pos += torch.bincount(idx[lb], minlength=bins).double()
        neg += torch.bincount(idx[~lb], minlength=bins).double()
    P, N = pos.sum(), neg.sum()
    neg_below = torch.cumsum(neg, 0) - neg
    return float(((pos * neg_below).sum() + 0.5 * (pos * neg).sum()) / (P * N))


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def config_name(a):
    parts = [a.adapter, f"L{'-'.join(map(str, a.blocks))}", f"fuse-{a.fusion}", f"p-{a.prompts}", f"e{a.epochs}"]
    if a.gamma0 != 0.01:
        parts.append(f"g{a.gamma0}")
    if a.heads != "both":
        parts.append(f"h-{a.heads}")
    if a.w_mem_loss:
        parts.append(f"mem{a.w_mem_loss}")
    if a.w_text_loss != 1.0:
        parts.append(f"txt{a.w_text_loss}")
    if a.fusion == "support" and a.w_text != 0.5:
        parts.append(f"wt{a.w_text}")
    if a.image_score != "max":
        parts.append(f"img-{a.image_score}")
    if a.det_pool != "mean":
        parts.append(f"dp-{a.det_pool}")
    if a.dual_memory:
        parts.append("dualmem")
    if a.eval_every:
        parts.append(f"curve{a.eval_every}")
    return "_".join(parts)


def run_one(bb, args, dataset, k, set_id, test_tok, meta, test_masks, text, block_idx, log):
    seg = dataset in SEG_DATASETS
    seed_all(set_id * 100 + k)
    sup_n, sup_a, overlap = support_set(dataset, k, set_id)
    rows = sup_a + sup_n
    imgs = torch.stack([load_img(r[0]) for r in rows])
    masks = torch.stack([load_mask(r[2], EVAL) for r in rows])
    labels0 = torch.tensor([r[1] for r in rows])
    t0 = time.time()
    a_img, a_mask, src = augment(imgs, masks)
    labels = labels0[src]
    with torch.no_grad():
        sup_tok = torch.cat([check(bb.tokens(norm(a_img[i:i + 32].to(DEV)), args.blocks), "support features").half().cpu()
                             for i in range(0, len(a_img), 32)])
    t_feat = time.time() - t0

    model = FSAD(len(args.blocks), bb.norm, bb.proj, text, mode=args.adapter, gamma0=args.gamma0).to(DEV)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    y_all = np.array(meta["labels"])

    def fuse_with(a, b, head, heads, fusion, w_text, stats):
        a, b = a.float(), b.float()
        if heads == "text":
            return a
        if heads == "memory":
            return b
        if fusion == "support":
            (ma, sa), (mb, sb) = stats[head]["text"], stats[head]["mem"]
            return w_text * (a - ma) / sa + (1 - w_text) * (b - mb) / sb
        # MVFA: min-max over the test set, then average (transductive)
        return 0.5 * (a - a.min()) / (a.max() - a.min() + 1e-12) + 0.5 * (b - b.min()) / (b.max() - b.min() + 1e-12)

    def img_from(m):
        flat = m.flatten(1)
        if args.image_score == "max":
            return flat.max(1).values.float()
        return flat.float().topk(max(1, flat.shape[1] // 100), dim=1).values.mean(1)

    def evaluate():
        """Score every test image with the current adapters; every scoring variant at once."""
        model.eval()
        normal_tok = sup_tok[labels == 0]
        mem_seg = memory_bank(model, normal_tok, "seg") if seg else None
        mem_det = memory_bank(model, normal_tok, "det")
        mem_frozen = frozen_bank(normal_tok) if (args.dual_memory and args.adapter != "none") else None
        stats = support_stats(model, sup_tok, src, labels, seg, det_pool=args.det_pool,
                              dual_memory=mem_frozen is not None)
        n = meta["n"]
        y = y_all
        det_text, det_mem = torch.zeros(n), torch.zeros(n)
        masks_eval = test_masks
        if seg:  # both maps are kept (fp16) so that every scoring variant can be read off one run
            seg_t = torch.zeros((n, EVAL, EVAL), dtype=torch.float16)
            seg_m = torch.zeros((n, EVAL, EVAL), dtype=torch.float16)
        t2 = time.time()
        for s in range(0, n, args.eval_batch):
            t = torch.from_numpy(np.ascontiguousarray(test_tok[s:s + args.eval_batch][:, block_idx]))
            check(t, "cached test features")
            o = score_batch(model, t, mem_seg, mem_det, seg, det_pool=args.det_pool, mem_frozen=mem_frozen)
            det_text[s:s + len(t)] = check(o["det_text"], "text scores").cpu()
            det_mem[s:s + len(t)] = check(o["det_mem"], "memory scores").cpu()
            if seg:
                seg_t[s:s + len(t)] = check(o["seg_text"], "text maps").half().cpu()
                seg_m[s:s + len(t)] = check(o["seg_mem"], "memory maps").half().cpu()
            del o
        t_eval = time.time() - t2
        if args.dev:  # evaluate on valid images that are not in the support set
            sup_paths = {r[0] for r in rows}
            keep = np.array([p not in sup_paths for p in meta["paths"]])
            y = y[keep]
            kt = torch.from_numpy(keep)
            det_text, det_mem = det_text[kt], det_mem[kt]
            if seg:
                seg_t, seg_m = seg_t[kt], seg_m[kt]
                masks_eval = test_masks[keep]
            n = int(keep.sum())
        if not torch.isfinite(det_mem).all() or not torch.isfinite(det_text).all() or det_mem.std() == 0:
            raise GPUFault(f"invalid scores for {dataset} k={k} set={set_id}; run not recorded")
        variants = {}
        schemes = {"text": ("text", "support", 0.5), "memory": ("memory", "support", 0.5),
                   "support": ("both", "support", args.w_text), "testminmax": ("both", "testminmax", 0.5)}
        if seg:
            gt = torch.from_numpy(masks_eval).bool().flatten()
            for name, (h, f, w) in schemes.items():
                m = torch.cat([fuse_with(seg_t[i:i + 256], seg_m[i:i + 256], "seg", h, "support", w, stats)
                               for i in range(0, n, 256)]) if f == "support" else fuse_with(seg_t, seg_m, "seg", h, f, w, stats)
                variants[f"pixel_{name}"] = hist_auc(m.flatten(), gt)
                variants[f"image_max_{name}"] = float(roc_auc_score(y, img_from(m).numpy()))
                del m
            del seg_t, seg_m
        for name, (h, f, w) in schemes.items():
            variants[f"image_det_{name}"] = float(roc_auc_score(y, fuse_with(det_text, det_mem, "det", h, f, w, stats).numpy()))
        del mem_seg, mem_det, mem_frozen
        if DEV.type == "mps":
            torch.mps.empty_cache()
        model.train()
        return variants, t_eval, n

    # optional learning curve on the evaluation split, used only to measure how much
    # selecting the epoch on test inflates results (MVFA's code does this); never for our numbers
    curve = {}
    def on_epoch(ep):
        if args.eval_every and ep % args.eval_every == 0 and ep < args.epochs:
            curve[ep] = evaluate()[0]

    t1 = time.time()
    train(model, sup_tok, a_mask, labels, src, seg, args.epochs if args.adapter != "none" else 0, args.lr, args.batch,
          w_text=args.w_text_loss, w_mem=args.w_mem_loss, on_epoch=on_epoch if args.eval_every else None)
    t_train = time.time() - t1
    variants, t_eval, n = evaluate()
    model.eval()
    if args.eval_every:
        curve[args.epochs] = variants

    res = {"dataset": dataset, "k": k, "set": set_id, "backbone": args.backbone, "config": config_name(args),
           "eval_split": "valid-minus-support" if args.dev else "test", "n_eval": n,
           "trainable_params": n_params, "sec_support_features": round(t_feat, 1), "sec_train": round(t_train, 1),
           "ms_per_test_image_heads": round(1000 * t_eval / n, 2), "support_in_test": overlap}
    if curve:
        res["curve"] = {str(e): v for e, v in sorted(curve.items())}
    res["variants"] = variants
    primary = "testminmax" if (args.fusion == "testminmax" and args.heads == "both") else (
        args.heads if args.heads != "both" else "support")
    if seg:
        res["pixel_auc"] = variants[f"pixel_{primary}"]
        res["image_auc"] = variants[f"image_max_{primary}"]
        res["image_auc_det_branch"] = variants[f"image_det_{primary}"]
    else:
        res["image_auc"] = variants[f"image_det_{primary}"]
    if args.adapter != "none":
        res["gates"] = [round(float(a.gamma.detach()), 4) for a in model.seg]
    bad = [m for m in ("image_auc", "pixel_auc") if m in res and not np.isfinite(res[m])]
    if bad:
        raise GPUFault(f"invalid scores for {dataset} k={k} set={set_id}: {bad}; run not recorded")
    del model, sup_tok
    if DEV.type == "mps":
        torch.mps.empty_cache()
    log.write(json.dumps(res) + "\n")
    log.flush()
    msg = f"pixel {100 * res['pixel_auc']:.2f} | " if seg else ""
    print(f"  {dataset:15s} k={k:<2d} set={set_id} {msg}image {100 * res['image_auc']:.2f} "
          f"(train {t_train:.0f}s, eval {t_eval:.0f}s)", flush=True)
    return res


def cmd_run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bb = Backbone(args.backbone)
    for dataset in args.datasets:
        split = "valid" if args.dev else "test"
        tok_p, meta_p, mask_p = cache_paths(args.backbone, dataset, split)
        if args.dev and not meta_p.exists():
            build_cache(args.backbone, dataset, CACHE_BLOCKS if dataset != "Chest" else [4, 8, 12],
                        split="valid", bb=bb)
        meta = json.loads(meta_p.read_text())
        missing = [b for b in args.blocks if b not in meta["blocks"]]
        if missing:
            raise SystemExit(f"blocks {missing} not cached for {dataset}; re-run `cache` with --blocks")
        block_idx = [meta["blocks"].index(b) for b in args.blocks]
        test_tok = np.load(tok_p, mmap_mode="r")
        test_masks = np.load(mask_p) if dataset in SEG_DATASETS else None
        text = bb.text(dataset, args.prompts)
        log_p = out / f"{args.backbone}__{config_name(args)}{'__dev' if args.dev else ''}.jsonl"
        done = set()
        if log_p.exists():
            done = {(r["dataset"], r["k"], r["set"]) for r in map(json.loads, log_p.read_text().splitlines())}
        with log_p.open("a") as log:
            for k in args.shots:
                for set_id in args.sets:
                    if (dataset, k, set_id) in done:
                        continue
                    for attempt in range(4):
                        try:
                            run_one(bb, args, dataset, k, set_id, test_tok, meta, test_masks, text, block_idx, log)
                            break
                        except GPUFault as e:
                            print(f"  GPU fault ({e}); retry {attempt + 1}/3 after freeing memory", flush=True)
                            if DEV.type == "mps":
                                torch.mps.empty_cache()
                            time.sleep(30)
                    else:
                        print(f"  FAILED {dataset} k={k} set={set_id}: GPU kept failing; nothing recorded", flush=True)
        del test_tok


def cmd_cache(args):
    for d in args.datasets:
        build_cache(args.backbone, d, args.blocks)


def cmd_table(args):
    import pandas as pd
    rows = []
    for p in Path(args.out).glob("*.jsonl"):
        rows += [json.loads(line) for line in p.read_text().splitlines()]
    if not rows:
        print("no results")
        return
    df = pd.DataFrame(rows)
    for metric in ("image_auc", "pixel_auc"):
        if metric not in df:
            continue
        sub = df.dropna(subset=[metric])
        g = sub.groupby(["backbone", "config", "dataset", "k"])[metric]
        t = (100 * g.mean()).round(2).astype(str) + "±" + (100 * g.std().fillna(0)).round(2).astype(str) + \
            " (n=" + g.size().astype(str) + ")"
        print(f"\n=== {metric} (mean ± std over support sets) ===")
        print(t.unstack("k").to_string())
    df.to_csv(Path(args.out) / "all_runs.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--backbone", choices=list(BACKBONES), default="openai")
    c.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    c.add_argument("--blocks", type=int, nargs="+", default=CACHE_BLOCKS)
    r = sub.add_parser("run")
    r.add_argument("--backbone", choices=list(BACKBONES), default="openai")
    r.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    r.add_argument("--shots", type=int, nargs="+", default=[2, 4, 8, 16])
    r.add_argument("--sets", type=int, nargs="+", default=[0, 1, 2])
    r.add_argument("--blocks", type=int, nargs="+", default=[4, 8, 12])
    r.add_argument("--adapter", choices=["shared", "dual", "none"], default="shared")
    r.add_argument("--fusion", choices=["support", "testminmax"], default="support")
    r.add_argument("--heads", choices=["both", "text", "memory"], default="both")
    r.add_argument("--prompts", choices=["mvfa", "medical"], default="mvfa")
    r.add_argument("--epochs", type=int, default=50)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--gamma0", type=float, default=0.01)
    r.add_argument("--w_text_loss", type=float, default=1.0, help="weight of MVFA's text-alignment losses")
    r.add_argument("--w_mem_loss", type=float, default=0.0, help="weight of the memory-consistency loss")
    r.add_argument("--w_text", type=float, default=0.5, help="weight of the text head in support-calibrated fusion")
    r.add_argument("--image_score", choices=["max", "top1"], default="max")
    r.add_argument("--dual_memory", action="store_true",
                   help="add the frozen-feature memory distance to the adapted-feature distance")
    r.add_argument("--det_pool", choices=["mean", "max"], default="mean",
                   help="pooling of patch memory distances into an image score (classification datasets)")
    r.add_argument("--eval_batch", type=int, default=32)
    r.add_argument("--out", default=str(ROOT.parent / "results_fsad"))
    r.add_argument("--eval_every", type=int, default=0,
                   help="also score the evaluation split every N epochs (epoch-selection study only)")
    r.add_argument("--dev", action="store_true",
                   help="evaluate on valid images outside the support set (for choosing settings; never the test split)")
    t = sub.add_parser("table")
    t.add_argument("--out", default=str(ROOT.parent / "results_fsad"))
    a = ap.parse_args()
    {"cache": cmd_cache, "run": cmd_run, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
