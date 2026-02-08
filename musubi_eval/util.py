import json
import logging
import random
import time
from typing import Any, Dict, Optional


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def sleep_backoff(attempt: int, base_sec: float, max_sec: float) -> None:
    # Exponential backoff with jitter
    delay = min(max_sec, base_sec * (2 ** max(0, attempt - 1)))
    delay *= random.uniform(0.8, 1.2)
    time.sleep(delay)


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def percentiles(values: list[float], ps: list[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {f"p{int(p)}": None for p in ps}
    vals = sorted(values)
    out: Dict[str, Optional[float]] = {}
    n = len(vals)
    for p in ps:
        if n == 1:
            out[f"p{int(p)}"] = vals[0]
            continue
        rank = (p / 100.0) * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        frac = rank - lo
        out[f"p{int(p)}"] = vals[lo] * (1 - frac) + vals[hi] * frac
    return out

