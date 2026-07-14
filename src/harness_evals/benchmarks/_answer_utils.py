"""Shared answer extraction utilities for benchmarks."""

from __future__ import annotations

import re


def extract_choice(response: str, choices: list[str] | None = None) -> str | None:
    """Extract a multiple-choice answer (A/B/C/D) from a model response.

    Tries several patterns in order of specificity:
    1. "The answer is (X)" or "the answer is X"
    2. Standalone letter at end of response
    3. First capital letter that matches valid choices
    """
    if choices is None:
        choices = ["A", "B", "C", "D"]

    text = response.strip()
    upper_choices = [c.upper() for c in choices]

    pattern = r"(?:the answer is|answer:)\s*\(?([A-Z])\)?"
    match = re.search(pattern, text, re.IGNORECASE)
    if match and match.group(1).upper() in upper_choices:
        return match.group(1).upper()

    pattern = r"\b([A-Z])\)?\.?\s*$"
    match = re.search(pattern, text)
    if match and match.group(1).upper() in upper_choices:
        return match.group(1).upper()

    for char in text:
        if char in upper_choices:
            return char

    return None


def extract_number(response: str) -> str | None:
    """Extract a numeric answer from a model response (for math benchmarks).

    Handles patterns like:
    - "The answer is 42"
    - "#### 42"
    - "= 42"
    - Negative numbers and decimals
    """
    text = response.strip()

    pattern = r"####\s*(-?[\d,]+\.?\d*)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).replace(",", "")

    pattern = r"(?:the answer is|answer:)\s*(-?[\d,]+\.?\d*)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")

    pattern = r"=\s*(-?[\d,]+\.?\d*)\s*$"
    match = re.search(pattern, text)
    if match:
        return match.group(1).replace(",", "")

    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def compute_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between prediction and reference strings."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
