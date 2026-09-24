"""Phase-1 probe: tokenizer fertility per locale and CPU timing for multilingual-e5-small.
Timing uses time.perf_counter (monotonic). Numbers go to phase1/time_encoder.out verbatim."""
import json, time, statistics, os, sys, random
import torch
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model

MODEL = "intfloat/multilingual-e5-small"
REV = "614241f622f53c4eeff9890bdc4f31cfecc418b3"  # sha from the HF API, 2026-09-24
LOCALES = ["en-US", "hi-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN"]
D = "data/1.1/data"
random.seed(0); torch.manual_seed(0)
print("torch", torch.__version__, "threads", torch.get_num_threads(), "interop", torch.get_num_interop_threads())

rows = {l: [json.loads(x) for x in open(f"{D}/{l}.jsonl", encoding="utf-8")] for l in LOCALES}
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
print("tokenizer", type(tok).__name__, "vocab", tok.vocab_size)

# 1. tokens per utterance per locale (train split, with the e5 "query: " prefix and special tokens)
for l in LOCALES:
    utts = [r["utt"] for r in rows[l] if r["partition"] == "train"]
    ids = tok(["query: " + u for u in utts], add_special_tokens=True)["input_ids"]
    n = sorted(len(x) for x in ids)
    unk = sum(x.count(tok.unk_token_id) for x in ids)
    chars = statistics.mean(len(u) for u in utts)
    words = statistics.mean(len(u.split()) for u in utts)
    print(f"tokens {l}: n={len(n)} mean={statistics.mean(n):.1f} median={n[len(n)//2]} p95={n[int(.95*len(n))]} max={n[-1]} "
          f"words/utt={words:.1f} tokens/word={(statistics.mean(n)-5)/words:.2f} chars/utt={chars:.1f} unk_tokens={unk}")

model = AutoModel.from_pretrained(MODEL, revision=REV)
total = sum(p.numel() for p in model.parameters())
emb = sum(p.numel() for n_, p in model.named_parameters() if "embeddings" in n_)
print(f"params total={total} embeddings={emb} non-embedding={total-emb}")

class Clf(torch.nn.Module):
    def __init__(self, enc, n_cls=54):
        super().__init__(); self.enc = enc; self.head = torch.nn.Linear(enc.config.hidden_size, n_cls)
    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1))  # mean pooling, as e5 is trained

cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"], lora_dropout=0.0, bias="none")
lora_enc = get_peft_model(model, cfg)
clf = Clf(lora_enc)
trainable = sum(p.numel() for p in clf.parameters() if p.requires_grad)
print(f"LoRA r=8 on query,value: trainable={trainable} ({100*trainable/total:.3f}% of encoder params; head included)")
opt = torch.optim.AdamW([p for p in clf.parameters() if p.requires_grad], lr=1e-3)

def batches(utts, bs):
    for i in range(0, len(utts), bs):
        yield tok(["query: " + u for u in utts[i:i+bs]], padding=True, truncation=True, max_length=64, return_tensors="pt")

def time_train(label, utts, steps=50, bs=32):
    clf.train(); ts = []
    it = batches(utts, bs)
    y = torch.randint(0, 54, (bs,))
    for s in range(steps + 3):
        b = next(it)
        t0 = time.perf_counter()
        logits = clf(b["input_ids"], b["attention_mask"])
        loss = torch.nn.functional.cross_entropy(logits, y[: logits.shape[0]])
        loss.backward(); opt.step(); opt.zero_grad()
        if s >= 3: ts.append(time.perf_counter() - t0)  # first 3 steps are warm-up, excluded
    print(f"train {label}: {steps} steps bs={bs} total={sum(ts):.1f}s per_step mean={statistics.mean(ts):.3f}s median={statistics.median(ts):.3f}s max={max(ts):.3f}s")
    return statistics.mean(ts)

def time_infer(label, utts, bs=64):
    clf.eval(); t0 = time.perf_counter()
    with torch.no_grad():
        for b in batches(utts, bs): clf(b["input_ids"], b["attention_mask"])
    dt = time.perf_counter() - t0
    print(f"infer {label}: {len(utts)} utts bs={bs} total={dt:.1f}s ({1000*dt/len(utts):.2f} ms/utt)")

train = {l: [r["utt"] for r in rows[l] if r["partition"] == "train"] for l in LOCALES}
mixed = [u for l in LOCALES for u in train[l][:1000]]; random.shuffle(mixed)
for rep in (1, 2):
    print(f"=== repeat {rep}")
    time_train("en-US", random.sample(train["en-US"], 53 * 32))
    time_train("6-locale mix", random.sample(mixed, 53 * 32))
    time_infer("en-US", random.sample(train["en-US"], 1000))
    time_infer("6-locale mix", random.sample(mixed, 1000))
