"""MASSIVE 1.1: the six locales, the pre-registered hold-out, and the sealed test split."""
import json
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "1.1" / "data"
RESULTS = ROOT / "results"
LOCALES = ("en-US", "hi-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN")
# Drawn 2026-09-24 with random.Random(20260924).sample(eligible, 6), eligible = 46 intents with >= 20
# en-US test examples; fixed in PREREGISTRATION.md before this file existed.
HELD_OUT = frozenset({"alarm_query", "email_query", "qa_stock", "recommendation_events",
                      "takeaway_order", "transport_query"})
_state = {"unsealed": False}


@lru_cache(maxsize=None)
def rows(locale):
    return tuple(json.loads(line) for line in (DATA / f"{locale}.jsonl").open(encoding="utf-8"))


def check_parallel():
    """Every locale has the same ids, each with the same partition and intent as en-US."""
    ref = {r["id"]: (r["partition"], r["intent"]) for r in rows("en-US")}
    for loc in LOCALES[1:]:
        other = {r["id"]: (r["partition"], r["intent"]) for r in rows(loc)}
        assert other == ref, f"{loc} is not parallel to en-US"
    return len(ref)


@lru_cache(maxsize=1)
def labels():
    """The 54 known intents, sorted; index = class id."""
    return tuple(sorted({r["intent"] for r in rows("en-US") if r["partition"] == "train"} - HELD_OUT))


def known(locale, partition):
    """Train or dev rows of known intents only. Test goes through test()."""
    assert partition in ("train", "dev"), "the test split is sealed: use test()"
    return [r for r in rows(locale) if r["partition"] == partition and r["intent"] not in HELD_OUT]


def unseal(reason):
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "UNSEAL.log").open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{reason}\n")
    _state["unsealed"] = True


def test(locale):
    """(known test rows, unknown test rows). Raises until unseal() has been called."""
    if not _state["unsealed"]:
        raise PermissionError("test split is sealed; call data.unseal(reason) once, after all tuning")
    t = [r for r in rows(locale) if r["partition"] == "test"]
    return [r for r in t if r["intent"] not in HELD_OUT], [r for r in t if r["intent"] in HELD_OUT]
