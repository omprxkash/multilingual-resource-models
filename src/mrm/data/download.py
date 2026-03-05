"""Download and cache datasets from HuggingFace Hub."""

import os
from pathlib import Path

import pandas as pd


def download_kinnews(output_dir: str = "data/raw") -> dict:
    """Download KINNEWS and KIRNEWS datasets from HuggingFace.

    Returns a dict with keys: kin_train, kin_test, kir_train, kir_test
    pointing to local CSV paths.
    """
    from datasets import load_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {}
    for lang, config in [("kin", "kinnews"), ("kir", "kirnews")]:
        ds = load_dataset("keleog/kinnews-kirnews", config)
        for split in ("train", "test"):
            key = f"{lang}_{split}"
            dest = out / f"{key}.csv"
            df = ds[split].to_pandas()
            df.to_csv(dest, index=False)
            paths[key] = dest
            print(f"  saved {key}: {len(df)} rows → {dest}")

    return paths


def download_afrisenti(
    languages: list = None,
    output_dir: str = "data/raw",
) -> dict:
    """Download AfriSenti-SemEval datasets for each language.

    Returns a dict keyed by lang_split (e.g. "kin_train") → Path.
    """
    from datasets import load_dataset

    if languages is None:
        languages = ["kin", "swa", "tso", "hau", "ibo", "yor"]

    # Map short codes to AfriSenti language names
    _lang_map = {
        "kin": "kinyarwanda",
        "swa": "swahili",
        "tso": "xitsonga",
        "hau": "hausa",
        "ibo": "igbo",
        "yor": "yoruba",
        "amh": "amharic",
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {}
    for code in languages:
        lang_name = _lang_map.get(code, code)
        try:
            ds = load_dataset("HausaNLP/AfriSenti-SemEval", lang_name)
            for split in ("train", "test"):
                if split not in ds:
                    continue
                key = f"{code}_{split}"
                dest = out / f"afrisenti_{key}.csv"
                df = ds[split].to_pandas()
                df.to_csv(dest, index=False)
                paths[key] = dest
                print(f"  saved afrisenti {key}: {len(df)} rows → {dest}")
        except Exception as exc:
            print(f"  warning: could not download afrisenti/{lang_name}: {exc}")

    return paths


def download_masakhaner(
    languages: list = None,
    output_dir: str = "data/raw",
) -> dict:
    """Download MasakhaNER dataset (optional, for NER extension)."""
    from datasets import load_dataset

    if languages is None:
        languages = ["kin", "swa", "hau", "ibo", "yor"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {}
    for lang in languages:
        try:
            ds = load_dataset("masakhane/masakhaner", lang)
            for split in ("train", "test", "validation"):
                if split not in ds:
                    continue
                key = f"{lang}_{split}"
                dest = out / f"masakhaner_{key}.csv"
                df = ds[split].to_pandas()
                df.to_csv(dest, index=False)
                paths[key] = dest
                print(f"  saved masakhaner {key}: {len(df)} rows → {dest}")
        except Exception as exc:
            print(f"  warning: could not download masakhaner/{lang}: {exc}")

    return paths
