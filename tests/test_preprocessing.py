"""Unit tests for the data preprocessing module."""

import pandas as pd
import pytest

from mrm.data.preprocessing import clean_text, remap_labels_zero_based, combine_title_content


def test_clean_text_removes_urls():
    text = "Check this http://example.com and www.test.org for news"
    result = clean_text(text)
    assert "http" not in result
    assert "www" not in result
    assert "news" in result


def test_clean_text_removes_html():
    text = "<b>Breaking</b> <i>news</i> today"
    result = clean_text(text)
    assert "<b>" not in result
    assert "breaking" in result
    assert "news" in result


def test_clean_text_lowercases():
    result = clean_text("KINYARWANDA Language")
    assert result == result.lower()


def test_clean_text_empty_string():
    assert clean_text("") == ""


def test_clean_text_non_string():
    assert clean_text(None) == ""
    assert clean_text(42) == ""


def test_remap_labels_zero_based_shifts():
    df = pd.DataFrame({"label": [1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14]})
    result = remap_labels_zero_based(df)
    assert result["label"].min() == 0
    assert result["label"].max() == len(result["label"].unique()) - 1


def test_remap_labels_drops_8_and_10():
    df = pd.DataFrame({"label": [1, 2, 8, 10, 3]})
    result = remap_labels_zero_based(df)
    # Only labels 1, 2, 3 should remain (mapped to 0, 1, 2)
    assert len(result) == 3
    assert set(result["label"]) == {0, 1, 2}


def test_remap_labels_contiguous():
    df = pd.DataFrame({"label": [1, 3, 5]})
    result = remap_labels_zero_based(df)
    # Should remap to 0, 1, 2 (contiguous)
    assert sorted(result["label"].unique()) == [0, 1, 2]


def test_combine_title_content():
    df = pd.DataFrame({"title": ["Breaking"], "content": ["News here"], "label": [1]})
    result = combine_title_content(df)
    assert "text" in result.columns
    assert "title" not in result.columns
    assert "Breaking" in result["text"].iloc[0]
    assert "News here" in result["text"].iloc[0]
