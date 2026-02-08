from typing import List

from musubi_eval.dataset import load_documents, load_queries
from musubi_eval.domain.models import Document, Query


class JsonlDatasetReader:
    def load_documents(self, path: str) -> List[Document]:
        return load_documents(path)

    def load_queries(self, path: str) -> List[Query]:
        return load_queries(path)
