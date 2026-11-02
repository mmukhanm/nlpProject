"""Общий корпус, обработка и метрики. Все разбиения сохраняются один раз."""
from __future__ import annotations

import hashlib
import html
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer, SnowballStemmer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from spacy.lang.en.stop_words import STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
NEGATIONS = {"no", "not", "never", "neither", "nor", "cannot"}
STOP = STOP_WORDS - NEGATIONS
LABELS = {0: "Негативный", 1: "Позитивный"}


def normalize(text: str) -> str:
    text = html.unescape(str(text)).lower().replace("’", "'")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)
    return " ".join(re.findall(r"[a-z]+", text))


def tokens(text: str, remove_stop: bool = True) -> list[str]:
    return [w for w in normalize(text).split() if not remove_stop or w not in STOP]


def clean(text: str) -> str:
    return " ".join(tokens(text))


def split_glued(text: str) -> str:
    """Отдельный эксперимент: автоматическое разделение может портить названия."""
    import wordninja
    return " ".join(wordninja.split(text))


@lru_cache(maxsize=1)
def spacy_model():
    import spacy
    return spacy.load("en_core_web_sm", disable=["ner"])


def morphology(texts: list[str]) -> pd.DataFrame:
    porter, snowball = PorterStemmer(), SnowballStemmer("english")
    rows = []
    # Raw sentences preserve the context needed by the POS tagger.
    for i, doc in enumerate(spacy_model().pipe(texts, batch_size=64)):
        for t in doc:
            if t.is_alpha:
                rows.append({"document": i, "word": t.text.lower(), "lemma": t.lemma_.lower(),
                             "pos": t.pos_, "morph": str(t.morph),
                             "porter": porter.stem(t.text.lower()),
                             "snowball": snowball.stem(t.text.lower())})
    return pd.DataFrame(rows)


def load_corpus() -> pd.DataFrame:
    path = ROOT / "data/raw/amazon_cells_labelled.txt"
    expected = "47003fc0a0d4840b00e96e715b6189bad09e7443a3da41c4cbe12ffc79f86ae3"
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Контрольная сумма корпуса не совпала. Проверьте источник.")
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"], quoting=3)
    if len(df) != 1000 or set(df.label) != {0, 1}:
        raise ValueError("Ожидались 1000 реальных Amazon-отзывов с метками 0/1")
    df["id"] = [f"amazon-{i:04d}" for i in range(len(df))]
    df["normalized"] = df.text.map(normalize)
    # Remove all conflicting copies before deduplication; preserve negation.
    conflicts = df.groupby("normalized").label.nunique()
    df = df[~df.normalized.isin(conflicts[conflicts > 1].index)]
    return df.drop_duplicates("normalized").reset_index(drop=True)


def prepare() -> pd.DataFrame:
    df = load_corpus()
    train, rest = train_test_split(df.index, test_size=.4, stratify=df.label, random_state=SEED)
    val, test = train_test_split(rest, test_size=.5, stratify=df.loc[rest, "label"], random_state=SEED)
    df["split"] = "train"
    df.loc[val, "split"], df.loc[test, "split"] = "validation", "test"
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "data/processed/reviews.csv", index=False)
    return df


def corpus() -> pd.DataFrame:
    p = ROOT / "data/processed/reviews.csv"
    return pd.read_csv(p) if p.exists() else prepare()


def metrics(y, pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y, pred)), "precision_macro": float(p),
            "recall_macro": float(r), "f1_macro": float(f), "n": len(y)}


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def mean_vectors(texts, vectors, dim: int) -> np.ndarray:
    rows = []
    for text in texts:
        words = tokens(text)
        known = [vectors[w] for w in words if w in vectors]
        rows.append(np.mean(known, axis=0) if known else np.zeros(dim))
    return np.asarray(rows, dtype=np.float32)
