# Sources, re-fetched 2026-09-24 09:17 IST by phase1/verify_sources.py

| Source | URL | HTTP | Bytes | Anchor string | Anchor check |
|---|---|---|---|---|---|
| MASSIVE dataset API (HF) | https://huggingface.co/api/datasets/AmazonScience/massive | 200 | 5,919 | license:cc-by-4.0 | FOUND  |
| MASSIVE dataset card (HF) | https://huggingface.co/datasets/AmazonScience/massive/resolve/main/README.md | 200 | 34,420 | MASSIVE 1.1 is a parallel dataset of > 1M utterances across 52 languages | FOUND  |
| MASSIVE 1.1 tarball (S3, from massive.py) | https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz | 200 | 40,251,390 | (binary; sha256 recorded separately) | n/a  |
| MASSIVE paper, FitzGerald et al. 2022/ACL 2023 | https://arxiv.org/pdf/2204.08582 | 200 | 542,788 | Table 8: Intent accuracy by language for our three models | FOUND p21 |
| MASSIVE paper, Table 3 | https://arxiv.org/pdf/2204.08582 | 200 | 542,788 | Table 3: Modeling results for (a) training runs on the full training dataset | FOUND p8 |
| IndicXTREME, Doddapaneni et al. 2023 | https://arxiv.org/pdf/2212.05409 | 200 | 1,218,146 | Table 19: Results on MASSIVE Intent Classification task. Metric: accuracy. | FOUND p21 |
| IndicXTREME, zero-shot protocol | https://arxiv.org/pdf/2212.05409 | 200 | 1,218,146 | We perform zero-shot evaluation by fine-tuning the model on English | FOUND p6 |
| LoRA, Hu et al. 2021 | https://arxiv.org/pdf/2106.09685 | 200 | 1,609,513 | We then scale | FOUND p4 |
| LoRA, Wq and Wv | https://arxiv.org/pdf/2106.09685 | 200 | 1,609,513 | we only apply LoRA to Wq and Wv in most experiments | FOUND p6 |
| Max-softmax baseline, Hendrycks and Gimpel 2017 | https://arxiv.org/pdf/1610.02136 | 200 | 211,205 | maximum softmax probabilities | FOUND p1 |
| Mahalanobis for intent OOD, Podolskiy et al. AAAI 2021 | https://arxiv.org/pdf/2101.03778 | 200 | 804,748 | RoBERTa equipped with the Mahalanobis dis | FOUND p4 |
| Threshold transfer, Insights 2022 | https://aclanthology.org/2022.insights-1.3/ | 200 | 42,692 | do not generalize well to the test-set | FOUND  |
| multilingual-e5-small API (HF) | https://huggingface.co/api/models/intfloat/multilingual-e5-small?expand[]=safetensors&expand[]=cardData | 200 | 528,782 | 117654272 | FOUND  |
| MTEB MassiveIntent task (main) | https://raw.githubusercontent.com/embeddings-benchmark/mteb/main/mteb/tasks/classification/multilingual/massive_intent_classification.py | 200 | 3,006 | class MassiveIntentClassification(AbsTaskClassification) | FOUND  |
| MTEB classification protocol (main) | https://raw.githubusercontent.com/embeddings-benchmark/mteb/main/mteb/abstasks/classification.py | 200 | 23,575 | samples_per_label: int = 8 | FOUND  |
| Multi3NLU++ card (HF) | https://huggingface.co/datasets/uoe-nlp/multi3-nlu/resolve/main/README.md | 200 | 5,903 | The dataset is Creative Commons Attribution 4.0 International (cc-by-4.0) | FOUND  |
| MuRIL config (HF) | https://huggingface.co/google/muril-base-cased/resolve/main/config.json | 200 | 411 | 197285 | FOUND  |
| XLM-R base API (HF) | https://huggingface.co/api/models/FacebookAI/xlm-roberta-base?expand[]=safetensors | 200 | 134 | 278885778 | FOUND  |
| IndicBERT v2 config (HF) | https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only/resolve/main/config.json | 200 | 639 | 250000 | FOUND  |
| PLANTED FALSE ANCHOR (must be MISSING) | https://arxiv.org/pdf/1610.02136 | 200 | 211,205 | LoRA beats every baseline on Tamil | MISSING  |
| PLANTED DEAD URL (must be FETCH FAILED) | https://huggingface.co/api/datasets/AmazonScience/this-dataset-does-not-exist-2026 | 401 | 0 | anything | FETCH FAILED  |

Both planted defects caught (false anchor MISSING, dead URL FETCH FAILED): True. Real sources failing: 0 of 19.
