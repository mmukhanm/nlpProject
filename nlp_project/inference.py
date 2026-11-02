"""Загрузка выбранной по validation модели; используется UI и CLI."""
import json
import joblib
import numpy as np
from .core import ROOT, LABELS, mean_vectors


class Predictor:
    def __init__(self):
        self.info = json.loads((ROOT / "reports/best_model.json").read_text())
        self.name = self.info["name"]
        if self.info["kind"] == "classical":
            self.bundle = joblib.load(ROOT / "models" / f"{self.name}.joblib")
        else:
            import torch
            from transformers import AutoTokenizer, BertModel, BertForSequenceClassification
            torch.set_num_threads(2)
            frozen = self.name == "bert_frozen_lr"
            path = ROOT / "models" / ("pretrained_bert" if frozen else self.name)
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            self.model = (BertModel if frozen else BertForSequenceClassification).from_pretrained(path, local_files_only=True)
            self.model.eval()
            if frozen: self.head = joblib.load(ROOT / "models/bert_frozen_lr.joblib")

    def predict(self, text: str):
        text = text.strip()
        if not text or not any(c.isalpha() for c in text):
            raise ValueError("Введите отзыв, содержащий слова.")
        if len(text) > 5000:
            raise ValueError("Максимальная длина — 5000 символов.")
        if not any("a" <= c.lower() <= "z" for c in text):
            raise ValueError("Модель обучена на английском. Введите отзыв на английском языке.")
        truncated = False
        if self.info["kind"] == "classical":
            x = [text] if self.bundle["kind"] == "sparse" else mean_vectors([text], self.bundle["vectors"], self.bundle["dim"])
            prob = float(self.bundle["model"].predict_proba(x)[0,1])
        else:
            import torch
            from .transformer import MAX_LENGTH, encode
            truncated = len(self.tokenizer(text)["input_ids"]) > MAX_LENGTH
            with torch.inference_mode():
                if self.name == "bert_frozen_lr":
                    prob = float(self.head.predict_proba(encode(self.model, self.tokenizer, [text]))[0,1])
                else:
                    batch = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
                    prob = float(torch.softmax(self.model(**batch).logits, dim=-1)[0,1])
        label = int(prob >= .5)
        return {"label": label, "sentiment": LABELS[label], "positive_probability": prob,
                "confidence": prob if label else 1-prob, "model": self.name, "truncated": truncated}
