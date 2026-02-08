from typing import Any, Dict, List, Protocol

from musubi_eval.domain.models import Document, Query


class SearchGateway(Protocol):
    def health(self) -> Dict[str, Any]:
        ...

    def documents_batch(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        ...

    def ingestion_start(self) -> Dict[str, Any]:
        ...

    def ingestion_get(self, job_id: str) -> Dict[str, Any]:
        ...

    def search(self, payload: Dict[str, Any]) -> List[str]:
        ...


class DatasetReader(Protocol):
    def load_documents(self, path: str) -> List[Document]:
        ...

    def load_queries(self, path: str) -> List[Query]:
        ...
