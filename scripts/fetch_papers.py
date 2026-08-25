"""
Download a small, deliberately INTERCONNECTED corpus of CV/ML papers from arXiv.

Why these eight papers specifically?
--------------------------------------
A knowledge graph is only interesting if the documents actually reference each
other. These eight are the canonical object-detection / transformer lineage, and
they form a dense citation-and-comparison web:

    Faster R-CNN  <--outperformed by--  Mask R-CNN, DETR
    YOLOv3        <--compared against-- basically everything
    ResNet        <--used as backbone by-- Faster R-CNN, Mask R-CNN, DETR
    Transformer   <--architecture used by-- ViT, DETR, Swin
    ViT           <--improved by--     Swin

That means Phase 4's graph will have real multi-hop paths in it, e.g.
    Swin --[improves on]--> ViT --[based on]--> Transformer --[proposed by]--> Vaswani
which is exactly the 3-hop chain that vector search structurally cannot follow.

All eight are open-access on arXiv. We fetch politely: one at a time, with a
delay between requests, and we skip anything already downloaded.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# arxiv_id -> human-readable filename. The filename matters: it becomes the
# citation string the LLM prints in Phase 6 ("[Source: detr.pdf, p.4]"), so we
# want something a human recognises, not "2005.12872.pdf".
PAPERS: dict[str, str] = {
    "1706.03762": "attention_is_all_you_need.pdf",   # Vaswani et al. - Transformer
    "1804.02767": "yolov3.pdf",                      # Redmon & Farhadi
    "2005.12872": "detr.pdf",                        # Carion et al. - DETR
    "2103.14030": "swin_transformer.pdf",            # Liu et al.
    "1506.01497": "faster_rcnn.pdf",                 # Ren et al.
    "1512.03385": "resnet.pdf",                      # He et al.
    "2010.11929": "vision_transformer.pdf",          # Dosovitskiy et al. - ViT
    "1703.06870": "mask_rcnn.pdf",                   # He et al.
}

# arXiv asks automated clients to identify themselves rather than use a default
# urllib user-agent (which they rate-limit aggressively).
USER_AGENT = "kg-rag-student-project/0.1 (educational use; contact via GitHub)"
DELAY_SECONDS = 3.0


def download_paper(arxiv_id: str, filename: str, dest_dir: Path) -> bool:
    """Download one arXiv PDF. Returns True if a new file was written."""
    dest = dest_dir / filename

    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  [skip]  {filename} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return False

    url = f"https://arxiv.org/pdf/{arxiv_id}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"  [FAIL]  {filename}: {exc}")
        return False

    # Sanity check: arXiv sometimes returns an HTML error page with a 200 status.
    # Every real PDF starts with the magic bytes b"%PDF".
    if not payload.startswith(b"%PDF"):
        print(f"  [FAIL]  {filename}: server returned non-PDF content ({len(payload)} bytes)")
        return False

    dest.write_bytes(payload)
    print(f"  [ok]    {filename}  ({len(payload) / 1e6:.1f} MB)  <- arXiv:{arxiv_id}")
    return True


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(PAPERS)} papers into {RAW_DIR}\n")

    downloaded = 0
    for index, (arxiv_id, filename) in enumerate(PAPERS.items()):
        if download_paper(arxiv_id, filename, RAW_DIR):
            downloaded += 1
            # Be a good citizen: pause between actual downloads, not between skips.
            if index < len(PAPERS) - 1:
                time.sleep(DELAY_SECONDS)

    present = sorted(RAW_DIR.glob("*.pdf"))
    total_mb = sum(p.stat().st_size for p in present) / 1e6
    print(f"\nDone. {downloaded} newly downloaded, {len(present)} PDFs now in data/raw/ ({total_mb:.1f} MB total)")


if __name__ == "__main__":
    main()
