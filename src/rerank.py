from typing import Sequence
from pydantic import PrivateAttr

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from ragatouille import RAGPretrainedModel

class ColBERTCompressor(BaseDocumentCompressor):
    """JaColBERTv2.5でrerankして、上位top-nを返す"""

    model_name: str
    candidate_pool: int = 10
    top_n: int = 3

    _model: RAGPretrainedModel = PrivateAttr()

    def model_post_init(self, _context) -> None:
        self._model = RAGPretrainedModel.from_pretrained(self.model_name)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        candidates = list(documents)[: self.candidate_pool]
        if not candidates:
            return []

        results = self._model.rerank(
            query=query,
            documents=[d.page_content for d in candidates],
            k=self.top_n,
        )
        
        out =[]
        for r in results:
            idx = r["result_index"]
            src = candidates[idx]
            out.append(Document(
                page_content=src.page_content,
                metadata={**src.metadata, "rerank_score": float(r["score"])},
            ))
        return out
