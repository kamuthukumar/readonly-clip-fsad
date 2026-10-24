"""Extract selected BMAD folders from the Hugging Face mirror without downloading the whole zip.

The mirror (datasets/code-dev05/BMAD, CC BY-NC-SA, same layout as the official
BMAD release) is one 21.8 GB zip. Few-shot experiments only need the valid and
test splits, so this script reads the zip index remotely and streams just the
requested members.

  python experiments/fetch_bmad.py --out ~/data/BMAD --datasets Brain Liver Retina_RESC \
      --splits valid test
"""

from __future__ import annotations

import argparse
import shutil
import time
import zipfile
from pathlib import Path

from huggingface_hub import HfFileSystem

REPO_FILE = "datasets/code-dev05/BMAD/bmad.zip"
ALL = ["Brain", "Liver", "Retina_RESC", "Retina_OCT2017", "Histopathology", "Chest"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+", default=ALL)
    ap.add_argument("--splits", nargs="+", default=["valid", "test"])
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    fs = HfFileSystem()
    with fs.open(REPO_FILE, "rb", block_size=64 * 2**20, cache_type="readahead") as f:
        z = zipfile.ZipFile(f)
        wanted = [i for i in z.infolist()
                  if not i.is_dir()
                  and i.filename.split("/")[0] in args.datasets
                  and len(i.filename.split("/")) > 1 and i.filename.split("/")[1] in args.splits]
        wanted.sort(key=lambda i: i.header_offset)  # sequential reads keep the read-ahead buffer useful
        total = sum(i.compress_size for i in wanted)
        print(f"{len(wanted)} files, {total / 1e9:.2f} GB to fetch into {out}")
        done, t0 = 0, time.time()
        for n, info in enumerate(wanted, 1):
            target = out / info.filename
            if target.exists() and target.stat().st_size == info.file_size:
                done += info.compress_size
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 2**20)
            done += info.compress_size
            if n % 500 == 0 or n == len(wanted):
                rate = done / max(time.time() - t0, 1e-6) / 1e6
                print(f"  {n}/{len(wanted)} files  {done / 1e9:.2f}/{total / 1e9:.2f} GB  {rate:.1f} MB/s", flush=True)


if __name__ == "__main__":
    main()
