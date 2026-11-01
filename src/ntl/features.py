"""Ручные фичи: стилометрия, формат, служебные слова, скелет."""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

PUNCT = set(".,;:!?—-–()[]{}\"'“”‘’…")
MARKDOWN = set("*_#`>|")
DISCOURSE = [
    "however", "moreover", "therefore", "thus", "ultimately", "consequently",
    "furthermore", "nevertheless", "in conclusion", "in summary", "overall",
    "importantly", "notably", "crucially", "essentially", "fundamentally",
    "it's important to note", "it is important to note", "delve", "tapestry",
    "realm", "foster", "intricate", "nuanced", "crucial", "pivotal", "profound",
]


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def stylometry(texts) -> tuple[np.ndarray, list[str]]:
    """~40 числовых признаков на текст."""
    rows = []
    for text in texts:
        words = re.findall(r"\S+", text)
        clean = [re.sub(r"[^\w'-]", "", w) for w in words]
        clean = [w for w in clean if w]
        wl = np.array([len(w) for w in clean], dtype=float) if clean else np.array([0.0])
        sents = _sentences(text)
        sl = np.array([len(s.split()) for s in sents], dtype=float) if sents else np.array([0.0])
        n_words = max(len(words), 1)
        n_chars = max(len(text), 1)
        lower = [w.lower() for w in clean]
        ttr = len(set(lower)) / max(len(lower), 1)
        hapax = sum(1 for w in set(lower) if lower.count(w) == 1) / max(len(set(lower)), 1)
        stop_rate = sum(w in ENGLISH_STOP_WORDS for w in lower) / max(len(lower), 1)
        punct = [c for c in text if c in PUNCT]
        md = sum(c in MARKDOWN for c in text)
        rows.append([
            len(text), n_words, len(sents), len(re.findall(r"\n\n", text)),
            wl.mean(), wl.std(), np.percentile(wl, 10), np.percentile(wl, 50),
            np.percentile(wl, 90), wl.max(),
            sl.mean(), sl.std(), sl.min(), sl.max(),
            text.count(",") / n_words, text.count(".") / n_words,
            text.count(";") / n_words, text.count(":") / n_words,
            text.count("—") / n_words, text.count("-") / n_words,
            text.count("(") / n_words, text.count(")") / n_words,
            text.count('"') / n_words, text.count("'") / n_words,
            text.count("“") / n_words, text.count("”") / n_words,
            text.count("’") / n_words, text.count("!") / n_words,
            text.count("?") / n_words, text.count("…") / n_words,
            sum(c.isupper() for c in text) / n_chars,
            sum(c.isdigit() for c in text) / n_chars,
            text.count("\n") / n_chars, md / n_chars,
            ttr, hapax, stop_rate,
            len(punct) / n_words,
            sum(w.isupper() and len(w) > 1 for w in clean) / max(len(clean), 1),
            sum(w[0].isupper() for w in clean if w) / max(len(clean), 1),
            float(np.mean([text.lower().count(d) for d in DISCOURSE])),
        ])
    names = [
        "chars", "words", "sents", "paragraphs",
        "wl_mean", "wl_std", "wl_p10", "wl_p50", "wl_p90", "wl_max",
        "sl_mean", "sl_std", "sl_min", "sl_max",
        "comma", "period", "semicolon", "colon", "emdash", "hyphen",
        "lparen", "rparen", "dquote", "squote", "ldquo", "rdquo", "rsquo",
        "excl", "quest", "ellipsis",
        "upper_rate", "digit_rate", "newline_rate", "markdown_rate",
        "ttr", "hapax", "stop_rate", "punct_rate", "allcaps", "title_start",
        "discourse",
    ]
    return np.array(rows, dtype=np.float64), names


def format_features(texts) -> tuple[np.ndarray, list[str]]:
    """Сырые счётчики форматирования и типографики."""
    rows = []
    for text in texts:
        n = max(len(text), 1)
        rows.append([
            text.count("\n"), text.count("\n\n"), text.count("*"), text.count("_"),
            text.count("#"), text.count("`"), text.count(">"), text.count("|"),
            text.count("—"), text.count("–"), text.count("-"), text.count("“"),
            text.count("”"), text.count("’"), text.count("…"), text.count("$"),
            text.count("("), text.count("["), text.count("•"), text.count("1."),
            sum(c.isupper() for c in text) / n,
            sum(c.isdigit() for c in text) / n,
            len(text) / max(len(text.split()), 1),
        ])
    names = [
        "nl", "nl2", "star", "underscore", "hash", "backtick", "gt", "pipe",
        "emdash", "endash", "hyphen", "ldquo", "rdquo", "rsquo", "ellipsis",
        "dollar", "lparen", "lbracket", "bullet", "numbered",
        "upper_rate", "digit_rate", "char_per_word",
    ]
    return np.array(rows, dtype=np.float64), names


def function_words(texts) -> np.ndarray:
    out = []
    for text in texts:
        toks = re.findall(r"[a-z']+", text.lower())
        out.append(" ".join(t for t in toks if t in ENGLISH_STOP_WORDS))
    return np.array(out)


def skeleton(texts) -> np.ndarray:
    """Первые буквы слов + пунктуация: темо-независимый «скелет»."""
    out = []
    for text in texts:
        parts = []
        for tok in re.findall(r"[A-Za-z']+|[^\sA-Za-z0-9]", text):
            if tok[0].isalpha():
                parts.append(tok[0].lower() if tok[0].islower() else tok[0])
            else:
                parts.append(tok)
        out.append(" ".join(parts))
    return np.array(out)
