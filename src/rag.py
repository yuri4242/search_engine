import yaml
import json
import lancedb

from sudachipy import Dictionary, SplitMode

from langchain_community.vectorstores import LanceDB
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever

from src.embeddings import E5Embeddings

def build_ensemble_retriever(config):
    """Vector とKeyword のhybrid検索"""

    # Embedding
    embedding = E5Embeddings(
        model_name=config["embedding"]["model"],
        encode_kwargs={"normalize_embeddings": config["embedding"]["normalize"]},
    )

    # Vector
    db = lancedb.connect(config["vectorstore"]["path"])
    vectorstore = LanceDB(connection=db, table_name=config["vectorstore"]["table_name"], embedding=embedding)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": config["vectorstore"]["retriever_k"]})

    # Keyword
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
    return ensemble_retriever


if __name__ == "__main__":
    config = yaml.safe_load(open("config/langchain_rag.yaml"))
    retriever = build_ensemble_retriever(config)
    results = retriever.invoke("RAGのチャンク分割")
    for d in results[:5]:
        print(d.metadata["article_id"], d.metadata["article_title"][:50])
