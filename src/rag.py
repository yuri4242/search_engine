import yaml
import lancedb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from typing import List

import json
from sudachipy import Dictionary, SplitMode
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from langchain.retrievers import EnsembleRetriever

config = yaml.safe_load(open("config/langchain_rag.yaml"))

class E5Embeddings(HuggingFaceEmbeddings):
    query_prefix: str="query: "
    passage_prefix: str="passage: "
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents([self.passage_prefix + t for t in texts])
    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(self.query_prefix + text)

embedding = E5Embeddings(
    model_name=config["embedding"]["model"],
    encode_kwargs={"normalize_embeddings": config["embedding"]["normalize"]},
)

db = lancedb.connect(config["vectorstore"]["path"])
vectorstore = LanceDB(connection=db, table_name=config["vectorstore"]["table_name"], embedding=embedding)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": config["vectorstore"]["retriever_k"]})

tokenizer = Dictionary().create()

def sudachi_preprocess(text: str) -> list[str]:
    morphemes = tokenizer.tokenize(text, mode=getattr(SplitMode, config["bm25"]["split_mode"]))
    return [m.normalized_form() for m in morphemes]

documents = []
with open(config["bm25"]["chunks_path"], encoding="utf-8") as f:
    for line in f:
        chunk = json.loads(line)
        documents.append(Document(
            page_content=chunk["text"],
            metadata={
                "article_id": chunk["article_id"],
                "chunk_index": chunk["chunk_index"],
                "article_title": chunk["article_title"],
                "article_url": chunk["article_url"],
            },
        ))

bm25_retriever = BM25Retriever.from_documents(
    documents,
    preprocess_func=sudachi_preprocess,
)
bm25_retriever.k = config["bm25"]["retriever_k"]

ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5],
)

results = ensemble_retriever.invoke("RAGのチャンク分割")
for d in results:
    print(d.metadata.get("article_id", "?"), d.page_content[:50])

"""
results = bm25_retriever.invoke("RAGのチャンク分割")
for d in results:
    print(d.metadata.get("article_id", "?"), d.page_content[:50])

results = retriever.invoke("RAGのチャンク分割")
for d in results:
    print(d.metadata)
    print(d.metadata.get("article_id", "?"), d.page_content[:50])
""" 
