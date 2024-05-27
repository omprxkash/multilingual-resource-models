"""Text cleaning, label remapping, and Word2Vec embedding builder."""

import re
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Labels 8 (Fashion) and 10 (Tourism) are dropped following the original paper
_DROP_LABELS_1INDEXED = {8, 10}


def clean_text(text: str, language: str = "kinyarwanda") -> str:
    """Remove noise from raw news text.

    Strips URLs, HTML tags, repeated punctuation, and normalises whitespace.
    For Kinyarwanda/Kirundi the same regex pipeline applies; language param
    is reserved for future stopword integration.
    """
    if not isinstance(text, str):
        return ""
    # URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # HTML
    text = re.sub(r"<[^>]+>", " ", text)
    # non-letter/digit characters (keep hyphens inside words)
    text = re.sub(r"[^\w\s\-]", " ", text, flags=re.UNICODE)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def remap_labels_zero_based(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Convert 1-indexed labels to 0-indexed and drop under-represented classes.

    The KINNEWS corpus uses labels 1–14.  Labels 8 (Fashion) and 10 (Tourism)
    are dropped per the original benchmark to keep class distributions stable.
    After dropping, labels are remapped to a contiguous 0-indexed range.
    """
    df = df.copy()
    # Drop the two sparse classes
    df = df[~df[label_col].isin(_DROP_LABELS_1INDEXED)].reset_index(drop=True)
    # Shift to 0-based
    df[label_col] = df[label_col] - 1
    # Re-index to contiguous range (gaps left by dropped labels)
    unique_sorted = sorted(df[label_col].unique())
    remap = {old: new for new, old in enumerate(unique_sorted)}
    df[label_col] = df[label_col].map(remap)
    return df


def combine_title_content(df: pd.DataFrame, sep: str = " [SEP] ") -> pd.DataFrame:
    """Concatenate title and content columns into a single 'text' column."""
    df = df.copy()
    df["text"] = df["title"].fillna("") + sep + df["content"].fillna("")
    return df.drop(columns=["title", "content"], errors="ignore")


def preprocess_kinnews(
    raw_path: str,
    output_path: str,
    language: str = "kinyarwanda",
    drop_sparse: bool = True,
) -> pd.DataFrame:
    """Full preprocessing pipeline for KINNEWS/KIRNEWS CSVs.

    Loads raw CSV, cleans text, remaps labels, saves cleaned version.
    """
    df = pd.read_csv(raw_path)
    df = combine_title_content(df)
    df["text"] = df["text"].apply(lambda t: clean_text(t, language))
    if drop_sparse:
        df = remap_labels_zero_based(df)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("preprocessed %s → %s (%d rows, %d classes)", raw_path, out, len(df), df["label"].nunique())
    return df


def build_word2vec_embeddings(
    train_path: str,
    test_path: str,
    output_txt: str,
    vector_size: int = 50,
    window: int = 5,
    min_count: int = 5,
) -> None:
    """Train skip-gram Word2Vec and export in GloVe text format.

    Combines train + test corpora so evaluation vocabulary is covered.
    Uses sg=1 (skip-gram) and hs=1 (hierarchical softmax) per the paper.
    """
    import nltk
    from gensim.models import Word2Vec

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    sentences = []
    for path in (train_path, test_path):
        df = pd.read_csv(path)
        text_col = "text" if "text" in df.columns else df.columns[-1]
        for row in df[text_col].dropna():
            sentences.append(nltk.word_tokenize(str(row).lower()))

    logger.info("training Word2Vec on %d sentences …", len(sentences))
    model = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,
        hs=1,
        workers=4,
        seed=42,
    )

    out = Path(output_txt)
    out.parent.mkdir(parents=True, exist_ok=True)
    vocab = model.wv.key_to_index
    with open(out, "w", encoding="utf-8") as f:
        for word in vocab:
            vec = " ".join(f"{v:.6f}" for v in model.wv[word])
            f.write(f"{word} {vec}\n")

    logger.info("saved %d vectors (%d-dim) → %s", len(vocab), vector_size, out)
