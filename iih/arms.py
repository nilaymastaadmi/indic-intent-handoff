"""The four arms: majority class, char TF-IDF + LR, frozen e5-small + LR, e5-small + LoRA."""
import json
import random
import time
from functools import lru_cache

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from iih.data import ROOT, labels

MODEL = "intfloat/multilingual-e5-small"
REV = "614241f622f53c4eeff9890bdc4f31cfecc418b3"   # pinned in phase 1 (HF API, 2026-09-24)
MAX_LEN, BATCH = 64, 32
CACHE, MODELS = ROOT / "cache", ROOT / "models"


def majority(train_y):
    return max(set(train_y), key=list(train_y).count)


def tfidf_lr(texts, y, C):
    # max_features caps memory for 6 scripts x 2-5 char grams (AMENDMENTS.md A1)
    return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                                         min_df=2, max_features=300_000),
                         LogisticRegression(C=C, max_iter=2000)).fit(texts, y)


def frozen_lr(emb, y, C):
    return LogisticRegression(C=C, max_iter=2000).fit(emb, y)


@lru_cache(maxsize=1)
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL, revision=REV)


def _batch(texts):
    return tokenizer()(["query: " + t for t in texts], padding=True, truncation=True,
                       max_length=MAX_LEN, return_tensors="pt")


class Clf(torch.nn.Module):
    """Encoder, mean pooling (as e5 is trained), linear head over the 54 known intents."""

    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.head = torch.nn.Linear(enc.config.hidden_size, len(labels()))

    def pooled(self, b):
        h = self.enc(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1)

    def forward(self, b):
        return self.head(self.pooled(b))


def build(lora, seed):
    torch.manual_seed(seed)
    enc = AutoModel.from_pretrained(MODEL, revision=REV)
    if lora:
        enc = get_peft_model(enc, LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"],
                                             lora_dropout=0.1, bias="none"))
    else:
        enc.requires_grad_(False)
    return Clf(enc)


def trainable(clf):
    return sum(p.numel() for p in clf.parameters() if p.requires_grad)


@torch.no_grad()
def infer(clf, texts, bs=64):
    """(softmax probabilities, pooled embeddings) as numpy arrays."""
    clf.eval()
    probs, pooled = [], []
    for i in range(0, len(texts), bs):
        p = clf.pooled(_batch(texts[i:i + bs]))
        probs.append(torch.softmax(clf.head(p), -1).numpy())
        pooled.append(p.numpy())
    return np.concatenate(probs), np.concatenate(pooled)


def frozen_embeddings(texts, key):
    """L2-normalised mean-pooled e5-small embeddings, cached on disk under `key`."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{key}.npy"
    if path.exists():
        emb = np.load(path)
        assert len(emb) == len(texts), (key, len(emb), len(texts))
        return emb
    _, emb = infer(build(False, 0), texts)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(path, emb)
    return emb


def train_lora(texts, y, lr, epochs, seed, log_path):
    random.seed(seed), np.random.seed(seed)
    clf = build(True, seed)
    params = [p for p in clf.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = epochs * ((len(texts) + BATCH - 1) // BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    y = torch.tensor(y)
    rng, step, t0 = np.random.default_rng(seed), 0, time.perf_counter()
    for epoch in range(epochs):
        clf.train()
        order = rng.permutation(len(texts))
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            loss = torch.nn.functional.cross_entropy(clf(_batch([texts[j] for j in idx])), y[idx])
            loss.backward()
            opt.step(), sched.step(), opt.zero_grad()
            step += 1
            if step % 100 == 0 or step == steps:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "of": steps, "epoch": epoch, "loss": round(loss.item(), 4),
                                        "elapsed_s": round(time.perf_counter() - t0, 1)}) + "\n")
    return clf


def save_trainable(clf, path):
    MODELS.mkdir(exist_ok=True)
    torch.save({k: v for k, v in clf.state_dict().items() if "lora_" in k or k.startswith("head.")}, path)


def load_lora(path):
    clf = build(True, 0)
    missing = clf.load_state_dict(torch.load(path), strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    return clf
