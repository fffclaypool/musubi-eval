import logging
from typing import Any, Dict, List, Optional

import requests

from musubi_eval.config import RetryConfig
from musubi_eval.util import sleep_backoff

logger = logging.getLogger(__name__)


class MusubiHttpClient:
    def __init__(self, base_url: str, timeout_sec: float, retry: RetryConfig):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.retry = retry
        self.session = requests.Session()

    def _request(
        self, method: str, path: str, json_body: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    json=json_body,
                    timeout=self.timeout_sec,
                )
                if 400 <= resp.status_code < 500:
                    resp.raise_for_status()
                if resp.status_code >= 500:
                    body = (resp.text or "").strip()
                    body_preview = body[:300]
                    message = (
                        f"server error: {resp.status_code}"
                        if not body_preview
                        else f"server error: {resp.status_code} body={body_preview}"
                    )
                    raise requests.HTTPError(message, response=resp)
                if resp.headers.get("content-type", "").startswith("application/json"):
                    return resp.json()
                return {"text": resp.text}
            except requests.RequestException as exc:
                if exc.response is not None and 400 <= exc.response.status_code < 500:
                    raise
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

    def ingestion_get(self, job_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/ingestion/jobs/{job_id}")

    def search(self, payload: Dict[str, Any]) -> List[str]:
        resp = self._request("POST", "/search", payload)
        return _extract_ids(resp)


def _extract_ids(search_response: Dict[str, Any]) -> List[str]:
    if isinstance(search_response, list):
        items = search_response
    else:
        items = (
            search_response.get("results")
            or search_response.get("documents")
            or search_response.get("hits")
            or []
        )
    ids: List[str] = []
    for item in items:
        if isinstance(item, dict) and "id" in item:
            ids.append(str(item["id"]))
        elif isinstance(item, str):
            ids.append(item)
    return ids
