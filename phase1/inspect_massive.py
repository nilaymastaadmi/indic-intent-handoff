"""Phase-1 probe: locales, split sizes, first rows, parallelism of ids across locales."""
import json, os, collections, sys
D = "data/1.1/data"
files = sorted(f for f in os.listdir(D) if f.endswith(".jsonl"))
print("locale files:", len(files))
print("locales:", " ".join(f[:-6] for f in files))
CHOSEN = ["en-US", "hi-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN"]
rows = {}
for loc in CHOSEN:
    path = f"{D}/{loc}.jsonl"
    rows[loc] = [json.loads(l) for l in open(path, encoding="utf-8")]
    c = collections.Counter(r["partition"] for r in rows[loc])
    print(loc, "bytes", os.path.getsize(path), "rows", len(rows[loc]), dict(c), "intents", len({r["intent"] for r in rows[loc]}))
print("keys:", sorted(rows["en-US"][0].keys()))
for loc in CHOSEN:
    print(f"--- first 3 rows {loc}")
    for r in rows[loc][:3]:
        print(json.dumps({k: r[k] for k in ("id", "locale", "partition", "intent", "utt")}, ensure_ascii=False))
# parallelism: same id -> same partition and same intent in every locale
base = {r["id"]: (r["partition"], r["intent"]) for r in rows["en-US"]}
for loc in CHOSEN[1:]:
    other = {r["id"]: (r["partition"], r["intent"]) for r in rows[loc]}
    same_ids = set(base) == set(other)
    mism = sum(1 for i in base if i in other and base[i] != other[i])
    print(f"{loc}: id sets equal to en-US: {same_ids}; ids {len(other)}; partition/intent mismatches: {mism}")
# intent counts per partition in en-US
for p in ("train", "dev", "test"):
    c = collections.Counter(r["intent"] for r in rows["en-US"] if r["partition"] == p)
    print(p, "min/max per intent", min(c.values()), max(c.values()), "n intents", len(c))
# duplicate utterance text within en-US train and across train/test
tr = collections.Counter(r["utt"].lower() for r in rows["en-US"] if r["partition"] == "train")
te = [r["utt"].lower() for r in rows["en-US"] if r["partition"] == "test"]
print("en-US test utts whose exact text appears in train:", sum(1 for u in te if u in tr), "of", len(te))
