import logging
import time
from typing import Any, Dict, List, Optional

import requests

from .config import RetryConfig
from .util import sleep_backoff

logger = logging.getLogger(__name__)


class MusubiClient:
    def __init__(self, base_url: str, timeout_sec: float, retry: RetryConfig):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.retry = retry
        self.session = requests.Session()

    def _request(self, method: str, path: str, json_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    json=json_body,
                    timeout=self.timeout_sec,
                )
                if resp.status_code >= 500:
                    raise requests.HTTPError(f"server error: {resp.status_code}")
                resp.raise_for_status()
                if resp.headers.get("content-type", "").startswith("application/json"):
                    return resp.json()
                return {"text": resp.text}
            except Exception as exc:
                if attempt >= self.retry.max_attempts:
                    logger.error("request failed after %s attempts: %s %s", attempt, method, url)
                    raise
                logger.warning("request failed (attempt %s): %s %s (%s)", attempt, method, url, exc)
                sleep_backoff(attempt, self.retry.base_backoff_sec, self.retry.max_backoff_sec)
        raise RuntimeError("unreachable")

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def documents_batch(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._request("POST", "/documents/batch", {"documents": documents})

    def ingestion_start(self) -> Dict[str, Any]:
        return self._request("POST", "/ingestion/jobs")

    def ingestion_last(self) -> Dict[str, Any]:
        return self._request("GET", "/ingestion/last")

    def ingestion_get(self, job_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/ingestion/jobs/{job_id}")

    def search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/search", payload)


class IngestionWaiter:
    def __init__(self, client: MusubiClient, poll_interval_sec: float, timeout_sec: float):
        self.client = client
        self.poll_interval_sec = poll_interval_sec
        self.timeout_sec = timeout_sec

    def wait_ready(self, job_id: str) -> Dict[str, Any]:
        start = time.time()
        while True:
            status = self.client.ingestion_get(job_id)
            state = str(status.get("state") or status.get("status") or "").lower()
            if state in {"ready", "completed", "done", "success"}:
                return status
            if state in {"failed", "error"}:
                raise RuntimeError(f"ingestion job failed: {status}")
            if time.time() - start > self.timeout_sec:
                raise TimeoutError("ingestion job timed out")
            time.sleep(self.poll_interval_sec)
