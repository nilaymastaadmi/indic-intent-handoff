"""Re-fetch every source cited in RESEARCH.md; record HTTP status, bytes, and whether the anchor
string is present in the fetched content (PDFs via pdftotext). Writes phase1/SOURCES.md.
The last row is a planted false anchor: the check must report it MISSING, or the checker is broken."""
import re, subprocess, sys, urllib.request, tempfile, os, datetime
UA = {"User-Agent": "Mozilla/5.0 (research provenance check; toshniwalnilay@gmail.com)"}
S = [  # (label, url, anchor, pdf_page or None)
 ("MASSIVE dataset API (HF)", "https://huggingface.co/api/datasets/AmazonScience/massive", "license:cc-by-4.0", None),
 ("MASSIVE dataset card (HF)", "https://huggingface.co/datasets/AmazonScience/massive/resolve/main/README.md", "MASSIVE 1.1 is a parallel dataset of > 1M utterances across 52 languages", None),
 ("MASSIVE 1.1 tarball (S3, from massive.py)", "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz", None, None),
 ("MASSIVE paper, FitzGerald et al. 2022/ACL 2023", "https://arxiv.org/pdf/2204.08582", "Table 8: Intent accuracy by language for our three models", 21),
 ("MASSIVE paper, Table 3", "https://arxiv.org/pdf/2204.08582", "Table 3: Modeling results for (a) training runs on the full training dataset", 8),
 ("IndicXTREME, Doddapaneni et al. 2023", "https://arxiv.org/pdf/2212.05409", "Table 19: Results on MASSIVE Intent Classification task. Metric: accuracy.", 21),
 ("IndicXTREME, zero-shot protocol", "https://arxiv.org/pdf/2212.05409", "We perform zero-shot evaluation by fine-tuning the model on English", None),
 ("LoRA, Hu et al. 2021", "https://arxiv.org/pdf/2106.09685", "We then scale", 4),
 ("LoRA, Wq and Wv", "https://arxiv.org/pdf/2106.09685", "we only apply LoRA to Wq and Wv in most experiments", None),
 ("Max-softmax baseline, Hendrycks and Gimpel 2017", "https://arxiv.org/pdf/1610.02136", "maximum softmax probabilities", 1),
 ("Mahalanobis for intent OOD, Podolskiy et al. AAAI 2021", "https://arxiv.org/pdf/2101.03778", "RoBERTa equipped with the Mahalanobis dis", 4),
 ("Threshold transfer, Insights 2022", "https://aclanthology.org/2022.insights-1.3/", "do not generalize well to the test-set", None),
 ("multilingual-e5-small API (HF)", "https://huggingface.co/api/models/intfloat/multilingual-e5-small?expand[]=safetensors&expand[]=cardData", "117654272", None),
 ("MTEB MassiveIntent task (main)", "https://raw.githubusercontent.com/embeddings-benchmark/mteb/main/mteb/tasks/classification/multilingual/massive_intent_classification.py", "class MassiveIntentClassification(AbsTaskClassification)", None),
 ("MTEB classification protocol (main)", "https://raw.githubusercontent.com/embeddings-benchmark/mteb/main/mteb/abstasks/classification.py", "samples_per_label: int = 8", None),
 ("Multi3NLU++ card (HF)", "https://huggingface.co/datasets/uoe-nlp/multi3-nlu/resolve/main/README.md", "The dataset is Creative Commons Attribution 4.0 International (cc-by-4.0)", None),
 ("MuRIL config (HF)", "https://huggingface.co/google/muril-base-cased/resolve/main/config.json", "197285", None),
 ("XLM-R base API (HF)", "https://huggingface.co/api/models/FacebookAI/xlm-roberta-base?expand[]=safetensors", "278885778", None),
 ("IndicBERT v2 config (HF)", "https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only/resolve/main/config.json", "250000", None),
 ("PLANTED FALSE ANCHOR (must be MISSING)", "https://arxiv.org/pdf/1610.02136", "LoRA beats every baseline on Tamil", None),
]
norm = lambda s: re.sub(r"\s+", " ", s)
cache, out = {}, []
for label, url, anchor, page in S:
    if url not in cache:
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read() if anchor is not None or not url.endswith(".tar.gz") else r.read()
                cache[url] = (r.status, body)
        except Exception as e:
            cache[url] = (getattr(e, "code", "ERR"), b"")
            m = re.match(r"https://raw.githubusercontent.com/([^/]+/[^/]+)/([^/]+)/(.+)", url)
            if m:  # structurally different transport: GitHub contents API, raw media type
                api = f"https://api.github.com/repos/{m[1]}/contents/{m[3]}?ref={m[2]}"
                try:
                    req = urllib.request.Request(api, headers={**UA, "Accept": "application/vnd.github.raw"})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        cache[url] = (f"{r.status} via api.github.com", r.read())
                except Exception as e2:
                    cache[url] = (f"ERR, api {getattr(e2, 'code', 'ERR')}", b"")
    status, body = cache[url]
    found, where = "n/a", ""
    if anchor is not None and not body:
        found = "FETCH FAILED"  # an empty fetch is a failure, never n/a
    elif anchor is not None and body:
        if body[:4] == b"%PDF":
            fd, p = tempfile.mkstemp(suffix=".pdf", dir="."); os.write(fd, body); os.close(fd)
            txt = subprocess.run(["pdftotext", p, "-"], capture_output=True).stdout.decode("utf-8", "replace")
            pages = txt.split("\f")
            hits = [i + 1 for i, pg in enumerate(pages) if norm(anchor) in norm(pg)]
            os.remove(p)
            found = "FOUND" if hits else "MISSING"; where = f"p{','.join(map(str, hits))}" if hits else ""
            if page and hits and page not in hits: where += f" (expected p{page})"
        else:
            found = "FOUND" if norm(anchor) in norm(body.decode("utf-8", "replace")) else "MISSING"
    out.append(f"| {label} | {url} | {status} | {len(body):,} | {anchor or '(binary; sha256 recorded separately)'} | {found} {where} |")
    print(out[-1], flush=True)
planted_ok = out[-1].rstrip(" |").endswith("MISSING")
real_missing = [o for o in out[:-1] if "| MISSING" in o or "| FETCH FAILED" in o]
with open("phase1/SOURCES.md", "w", encoding="utf-8") as f:
    f.write(f"# Sources, re-fetched {datetime.datetime.now():%Y-%m-%d %H:%M} IST by phase1/verify_sources.py\n\n")
    f.write("| Source | URL | HTTP | Bytes | Anchor string | Anchor check |\n|---|---|---|---|---|---|\n")
    f.write("\n".join(out) + "\n\n")
    f.write(f"Planted false anchor reported MISSING: {planted_ok}. Real sources with a missing anchor: {len(real_missing)} of {len(out)-1}.\n")
print("planted false anchor caught:", planted_ok, "| real missing:", len(real_missing), "of", len(out) - 1)
sys.exit(0 if planted_ok and not real_missing else 1)
