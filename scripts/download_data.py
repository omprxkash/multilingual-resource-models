#!/usr/bin/env python
"""Standalone script to download all project datasets from HuggingFace."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mrm.data.download import download_kinnews, download_afrisenti, download_masakhaner
from mrm.data.preprocessing import preprocess_kinnews


def parse_args():
    p = argparse.ArgumentParser(description="Download project datasets")
    p.add_argument("--output-dir", default="data/raw", help="Raw data directory")
    p.add_argument("--processed-dir", default="data/processed", help="Processed data directory")
    p.add_argument("--skip-ner", action="store_true", help="Skip MasakhaNER download")
    p.add_argument(
        "--languages",
        nargs="+",
        default=["kin", "swa", "hau", "ibo", "yor"],
        help="AfriSenti language codes",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("=== downloading KINNEWS / KIRNEWS ===")
    kinnews_paths = download_kinnews(output_dir=args.output_dir)

    print("\n=== preprocessing KINNEWS / KIRNEWS ===")
    for lang in ("kin", "kir"):
        for split in ("train", "test"):
            key = f"{lang}_{split}"
            if key in kinnews_paths:
                preprocess_kinnews(
                    raw_path=str(kinnews_paths[key]),
                    output_path=f"{args.processed_dir}/{key}.csv",
                    language="kinyarwanda" if lang == "kin" else "kirundi",
                )

    print("\n=== downloading AfriSenti ===")
    download_afrisenti(languages=args.languages, output_dir=args.output_dir)

    if not args.skip_ner:
        print("\n=== downloading MasakhaNER ===")
        download_masakhaner(languages=args.languages, output_dir=args.output_dir)

    print("\ndone — data is ready in", args.output_dir)


if __name__ == "__main__":
    main()
