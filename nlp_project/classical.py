"""Недели 1–6: статистика, морфология, векторы и классификация."""
from __future__ import annotations

from collections import Counter
import gzip
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import FastText, Word2Vec
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud

from .core import ROOT, SEED, clean, corpus, mean_vectors, metrics, morphology, save_json, split_glued, tokens


def vectorizer(kind="tfidf"):
    cls = TfidfVectorizer if kind == "tfidf" else CountVectorizer
    return cls(preprocessor=clean, ngram_range=(1, 2), min_df=1, max_features=12000)


def glove_vectors(vocabulary):
    """Официальные GloVe-50, загрузка через gensim-data (с проверкой checksum)."""
    import gensim.downloader as api
    path = api.load("glove-wiki-gigaword-50", return_path=True)
    found = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            word, _, tail = line.partition(" ")
            if word in vocabulary:
                found[word] = np.fromstring(tail, sep=" ", dtype=np.float32)
    return found


def run():
    started = time.monotonic()
    report, figures, models = ROOT / "reports", ROOT / "reports/figures", ROOT / "models"
    for p in (report, figures, models): p.mkdir(parents=True, exist_ok=True)
    df = corpus()
    train, val, test = [df[df.split == s] for s in ("train", "validation", "test")]
    stats = {"raw_documents": 1000, "unique_documents": len(df),
             "words": sum(len(tokens(t, False)) for t in df.text),
             "classes": df.label.value_counts().to_dict(), "splits": df.split.value_counts().to_dict(),
             "language": "English", "source": "UCI Sentiment Labelled Sentences / Amazon Cells"}
    save_json(report / "corpus_stats.json", stats)
    pd.DataFrame({"before": df.text.head(20), "after": df.text.head(20).map(clean)}).to_csv(report / "preprocessing.csv", index=False)
    pd.DataFrame([{"input": s, "segmented": split_glued(s)} for s in
                  ["thebatteryisnotgood", "soundqualityisexcellent", "thisphonedoesnotwork"]]).to_csv(report / "glued_text.csv", index=False)
    morph = morphology(df.text.tolist())
    morph.to_csv(report / "morphology.csv", index=False)
    for column in ("word", "lemma"):
        morph[column].value_counts().head(50).rename_axis(column).reset_index(name="count").to_csv(report / f"top50_{column}.csv", index=False)
    morph.drop_duplicates("word").head(100).to_csv(report / "stem_vs_lemma.csv", index=False)
    morph.pos.value_counts().plot.bar(color="#375aa5", title="Part of speech / Amazon reviews")
    plt.tight_layout(); plt.savefig(figures / "pos.png", dpi=160); plt.close()
    WordCloud(width=1000, height=500, background_color="white", random_state=SEED).generate(" ".join(map(clean, train.text))).to_file(str(figures / "wordcloud.png"))
    for name in ("bow", "tfidf"):
        vec = vectorizer(name)
        x = vec.fit_transform(train.text)
        scores = np.asarray(x.sum(axis=0)).ravel()
        top = np.argsort(scores)[-20:][::-1]
        terms = pd.DataFrame({"term": vec.get_feature_names_out()[top], "weight": scores[top]})
        terms.to_csv(report / f"{name}_terms.csv", index=False)
        terms.iloc[::-1].plot.barh(x="term", y="weight", legend=False, title=name.upper(), figsize=(8, 6))
        plt.tight_layout(); plt.savefig(figures / f"{name}_terms.png", dpi=160); plt.close()

    sentences = [tokens(t) for t in train.text]
    vocabulary = set(w for sent in sentences for w in sent)
    w2v = Word2Vec(sentences, vector_size=50, window=5, min_count=1, workers=1, seed=SEED, epochs=40, sg=1)
    ft = FastText(sentences, vector_size=50, window=5, min_count=1, workers=1, seed=SEED, epochs=40, sg=1, bucket=20000)
    w2v.save(str(models / "word2vec.model")); ft.save(str(models / "fasttext.model"))
    glove = glove_vectors(vocabulary)
    joblib.dump(glove, models / "glove_subset.joblib")
    embeddings = {"word2vec": w2v.wv, "fasttext": ft.wv, "glove": glove}
    examples = {}
    for name, vectors in embeddings.items():
        examples[name] = {}
        for word in ("battery", "phone", "sound", "good", "bad"):
            if word not in vectors: continue
            keys = sorted(vocabulary & set(vectors.key_to_index)) if name != "glove" else sorted(glove)
            mat = np.array([vectors[k] for k in keys]); q = vectors[word]
            sim = mat @ q / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-12)
            order = [i for i in np.argsort(sim)[::-1] if keys[i] != word][:5]
            examples[name][word] = [(keys[i], float(sim[i])) for i in order]
        if name != "glove":
            examples[name]["arithmetic: good + phone - bad"] = vectors.most_similar(positive=["good", "phone"], negative=["bad"], topn=5)
        else:
            q = vectors["good"] + vectors["phone"] - vectors["bad"]
            mat = np.array([vectors[k] for k in keys])
            scores = mat @ q / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-12)
            examples[name]["arithmetic: good + phone - bad"] = [(keys[i], float(scores[i])) for i in np.argsort(scores)[-5:][::-1]]
    save_json(report / "semantic_examples.json", examples)
    save_json(report / "embedding_coverage.json", {name: {"vocab": len(vocabulary), "known": sum(w in v for w in vocabulary)} for name, v in embeddings.items()})

    rows, predictions, estimators = [], [], {}
    candidates = {
        "bow_nb": Pipeline([("vec", vectorizer("bow")), ("clf", MultinomialNB())]),
        "tfidf_lr": Pipeline([("vec", vectorizer()), ("clf", LogisticRegression(C=2, max_iter=1000, random_state=SEED))]),
        "tfidf_nb": Pipeline([("vec", vectorizer()), ("clf", MultinomialNB())]),
    }
    for name, clf in candidates.items():
        clf.fit(train.text, train.label)
        estimators[name] = {"kind": "sparse", "model": clf}
    for name, vectors in embeddings.items():
        x = mean_vectors(train.text, vectors, 50)
        clf = LogisticRegression(C=2, max_iter=1000, random_state=SEED).fit(x, train.label)
        estimators[f"{name}_lr"] = {"kind": "embedding", "model": clf, "vectors": vectors, "dim": 50}
        if name == "word2vec":
            xy = PCA(n_components=2, random_state=SEED).fit_transform(x)
            plt.scatter(xy[:, 0], xy[:, 1], c=train.label, cmap="coolwarm", alpha=.6, s=14)
            plt.title("Mean Word2Vec: PCA (train)"); plt.tight_layout()
            plt.savefig(figures / "embedding_pca.png", dpi=160); plt.close()
    for name, bundle in estimators.items():
        for split, part in (("validation", val), ("test", test)):
            x = part.text if bundle["kind"] == "sparse" else mean_vectors(part.text, bundle["vectors"], 50)
            prob = bundle["model"].predict_proba(x)[:, 1]
            pred = (prob >= .5).astype(int)
            rows.append({"model": name, "split": split, **metrics(part.label, pred)})
            predictions.extend({"id": id_, "model": name, "split": split, "text": text, "label": int(y), "prediction": int(p), "positive_probability": float(pr)} for id_, text, y, p, pr in zip(part.id, part.text, part.label, pred, prob))
        joblib.dump(bundle, models / f"{name}.joblib")
    result = pd.DataFrame(rows)
    result.to_csv(report / "classical_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(report / "classical_predictions.csv", index=False)
    best = result[result.split == "validation"].sort_values(["f1_macro", "model"], ascending=[False, True]).iloc[0]
    save_json(report / "best_model.json", {"name": best.model, "kind": "classical", "selection": "validation macro-F1", "validation_f1": float(best.f1_macro)})
    print(result.to_string(index=False), flush=True)
    print(f"Classical workflow completed in {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__": run()
