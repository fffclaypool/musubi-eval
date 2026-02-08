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
    base_delay = min(max_sec, base_sec * (2 ** max(0, attempt - 1)))
    jittered_delay = base_delay * random.uniform(0.8, 1.2)
    time.sleep(jittered_delay)


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def percentiles(values: list[float], ps: list[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {f"p{int(p)}": None for p in ps}
    sorted_vals = sorted(values)
    count = len(sorted_vals)

    def percentile_value(p: float) -> float:
        if count == 1:
            return sorted_vals[0]
        rank_value = (p / 100.0) * (count - 1)
        lo_index = int(rank_value)
        hi_index = min(lo_index + 1, count - 1)
        frac = rank_value - lo_index
        lo_val = sorted_vals[lo_index]
        hi_val = sorted_vals[hi_index]
        return lo_val * (1 - frac) + hi_val * frac

    return {f"p{int(p)}": percentile_value(p) for p in ps}
