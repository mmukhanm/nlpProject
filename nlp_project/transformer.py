"""Недели 8–11. Настоящий pretrained BERT; нет случайной подмены модели."""
from __future__ import annotations

import argparse
import copy
import os
import random
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertModel, BertForSequenceClassification

from .classical import vectorizer
from .core import ROOT, SEED, corpus, metrics, save_json

MODEL_ID = "prajjwal1/bert-tiny"
MAX_LENGTH = 96
BATCH_SIZE = 16
EPOCHS = 4


def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def download_model():
    local = ROOT / "models/pretrained_bert"
    if (local / "config.json").exists() and any((local / n).exists() for n in ("model.safetensors", "pytorch_model.bin")):
        return local
    path = snapshot_download(MODEL_ID, allow_patterns=["config.json", "vocab.txt", "tokenizer*", "special_tokens_map.json", "model.safetensors", "pytorch_model.bin"])
    import shutil
    local.mkdir(parents=True, exist_ok=True)
    for p in __import__("pathlib").Path(path).iterdir():
        if p.is_file(): shutil.copy2(p, local / p.name)
    save_json(ROOT / "reports/bert_source.json", {"model": MODEL_ID, "revision": __import__("pathlib").Path(path).name,
              "purpose": "2-layer pretrained English BERT, 128 hidden; CPU educational experiment"})
    return local


class ReviewDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.enc = tokenizer(frame.text.tolist(), padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        self.labels = torch.tensor(frame.label.tolist(), dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return {**{k: v[i] for k, v in self.enc.items()}, "labels": self.labels[i]}


@torch.inference_mode()
def encode(model, tokenizer, texts):
    model.eval()
    result = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = tokenizer(list(texts[start:start+BATCH_SIZE]), padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        result.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).numpy())
    return np.concatenate(result)


@torch.inference_mode()
def probabilities(model, loader):
    model.eval()
    return np.concatenate([torch.softmax(model(**{k:v for k,v in b.items() if k != "labels"}).logits, dim=-1)[:, 1].numpy() for b in loader])


def fine_tune(path, tokenizer, train, val, lr, fraction, name, output_dir=None):
    seed_all()
    if fraction < 1:
        idx, _ = train_test_split(np.arange(len(train)), train_size=fraction, stratify=train.label, random_state=SEED)
        train = train.iloc[idx]
    train_loader = DataLoader(ReviewDataset(train, tokenizer), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(ReviewDataset(val, tokenizer), batch_size=BATCH_SIZE)
    model = BertForSequenceClassification.from_pretrained(path, num_labels=2, attn_implementation="eager")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    best_score, best_state, history = -1, None, []
    started = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        model.train(); total = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += loss.item() * len(batch["labels"])
        prob = probabilities(model, val_loader)
        score = metrics(val.label, prob >= .5)
        history.append({"model": name, "epoch": epoch, "lr": lr, "fraction": fraction, "train_n": len(train), "train_loss": total/len(train), **score})
        print(f"{name}: epoch {epoch}, loss={total/len(train):.4f}, validation F1={score['f1_macro']:.4f}", flush=True)
        if score["f1_macro"] > best_score:
            best_score = score["f1_macro"]
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    out = (ROOT / "models" if output_dir is None else output_dir) / name
    model.save_pretrained(out); tokenizer.save_pretrained(out)
    save_json(out / "training_config.json", {"lr": lr, "fraction": fraction, "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
             "epochs": EPOCHS, "seed": SEED, "selected_by": "validation macro-F1, earliest epoch on tie", "seconds": time.monotonic()-started})
    return model, history


def attention_experiment(encoder, tokenizer, val, train):
    selected = val.head(16).copy()
    z = encode(encoder, tokenizer, selected.text.tolist())
    tfidf = vectorizer().fit(train.text)
    c1, c2 = cosine_similarity(tfidf.transform(selected.text)), cosine_similarity(z)
    selected[["id", "text", "label"]].to_csv(ROOT / "reports/attention_texts.csv", index=False)
    rows = [{"id_a": selected.id.iloc[i], "id_b": selected.id.iloc[j], "tfidf_cosine": c1[i,j], "bert_cosine": c2[i,j]} for i in range(16) for j in range(i+1,16)]
    pd.DataFrame(rows).to_csv(ROOT / "reports/representation_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m, title in zip(axes, (c1, c2), ("TF-IDF", "Frozen BERT / mean pooling")):
        im = ax.imshow(m, vmin=0, vmax=1); ax.set_title(title); fig.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(ROOT / "reports/figures/representation_comparison.png", dpi=160); plt.close()
    batch = tokenizer(selected.text.iloc[0], return_tensors="pt", truncation=True, max_length=40)
    with torch.inference_mode(): out = encoder(**batch, output_attentions=True)
    weights = out.attentions[-1][0].mean(0).numpy()
    pieces = tokenizer.convert_ids_to_tokens(batch["input_ids"][0])
    pd.DataFrame(weights, index=pieces, columns=pieces).to_csv(ROOT / "reports/attention_matrix.csv")
    fig, ax = plt.subplots(figsize=(10, 8)); im = ax.imshow(weights, cmap="magma")
    ax.set_xticks(range(len(pieces)), pieces, rotation=90); ax.set_yticks(range(len(pieces)), pieces)
    ax.set_title("Pretrained BERT: last layer, mean over heads (query → key)")
    fig.colorbar(im, ax=ax); plt.tight_layout(); plt.savefig(ROOT / "reports/figures/attention.png", dpi=160); plt.close()


def run():
    seed_all()
    df = corpus()
    train, val, test = [df[df.split == s].reset_index(drop=True) for s in ("train", "validation", "test")]
    path = download_model()
    tokenizer = AutoTokenizer.from_pretrained(path)
    encoder = BertModel.from_pretrained(path, attn_implementation="eager")
    attention_experiment(encoder, tokenizer, val, train)
    embeddings = {s: encode(encoder, tokenizer, part.text.tolist()) for s, part in (("train", train), ("validation", val), ("test", test))}
    frozen = LogisticRegression(C=1, max_iter=1500, random_state=SEED).fit(embeddings["train"], train.label)
    joblib.dump(frozen, ROOT / "models/bert_frozen_lr.joblib")
    rows, predictions, history = [], [], []

    def record(name, split, part, prob):
        pred = (prob >= .5).astype(int)
        rows.append({"model": name, "split": split, **metrics(part.label, pred)})
        predictions.extend({"id": id_, "model": name, "split": split, "text": text, "label": int(y), "prediction": int(p), "positive_probability": float(pr)} for id_, text, y, p, pr in zip(part.id, part.text, part.label, pred, prob))

    for split, part in (("validation", val), ("test", test)):
        record("bert_frozen_lr", split, part, frozen.predict_proba(embeddings[split])[:, 1])
    # One factor at a time: learning rate on full train, data fraction at fixed lr.
    experiments = [(2e-5, 1., "bert_lr2e5_full"), (3e-5, 1., "bert_lr3e5_full"),
                   (5e-5, 1., "bert_lr5e5_full"), (3e-5, .5, "bert_lr3e5_half"), (3e-5, .75, "bert_lr3e5_75")]
    for lr, fraction, name in experiments:
        model, hist = fine_tune(path, tokenizer, train, val, lr, fraction, name)
        history.extend(hist)
        # Test is scored only for reporting; never used for checkpoint/model choice.
        for split, part in (("validation", val), ("test", test)):
            record(name, split, part, probabilities(model, DataLoader(ReviewDataset(part, tokenizer), batch_size=BATCH_SIZE)))
        pd.DataFrame(history).to_csv(ROOT / "reports/bert_training_history.csv", index=False)
        pd.DataFrame(rows).to_csv(ROOT / "reports/bert_metrics.csv", index=False)
        pd.DataFrame(predictions).to_csv(ROOT / "reports/bert_predictions.csv", index=False)
    pd.DataFrame(history).pivot(index="epoch", columns="model", values="f1_macro").plot(marker="o", figsize=(10, 5), title="Validation macro-F1 / epoch")
    plt.ylabel("Macro-F1"); plt.tight_layout(); plt.savefig(ROOT / "reports/figures/bert_learning_curves.png", dpi=160); plt.close()
    lengths = [len(tokenizer(t)["input_ids"]) for t in df.text]
    save_json(ROOT / "reports/token_lengths.json", {"max_length": MAX_LENGTH, "truncated_documents": sum(n > MAX_LENGTH for n in lengths), "longest": max(lengths)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__": run()
