import yaml
import json
import lancedb
import argparse
import time

from sudachipy import Dictionary, SplitMode

from langchain_community.vectorstores import LanceDB
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever

from src.embeddings import E5Embeddings
from src.rerank import ColBERTCompressor
from src.llm import build_llm

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

    compressor = ColBERTCompressor(
        model_name=config["reranker"]["model"],
        candidate_pool=config["reranker"]["candidate_pool"],
        top_n=config["reranker"]["top_n"],
    )
    return ContextualCompressionRetriever(
        base_retriever=ensemble_retriever,
        base_compressor=compressor,
    )

def format_docs(docs):
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

def build_rag_chain(config):
    retriever = build_ensemble_retriever(config)
    llm = build_llm(config)
    template = (
            f"{config['prompt']['system'].strip()}\n\n"
            "{context}\n\n"
            "質問: {question}\n"
            "回答: "
            )
    prompt = PromptTemplate.from_template(template)

    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt
        | llm
        | StrOutputParser()
    )

    return (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()}).assign(answer=answer_chain)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG質疑応答")
    parser.add_argument("--query", default="RAGのチャンク分割", help="質問")
    args = parser.parse_args()

    config = yaml.safe_load(open("config/langchain_rag.yaml"))
    
    print("初期化中...")
    init_start = time.time()
    chain = build_rag_chain(config)
    print(f"初期化完了 ({time.time() - init_start:.1f}秒) \n")

    print(f"[質問] {args.query}")
    print("-" * 70)

    invoke_start = time.time()
    result = chain.invoke(args.query)
    print(f"\n[回答] ({time.time() - invoke_start:.1f}秒) \n")
    print(result["answer"].strip())

    print("\n根拠:")
    for i, d in enumerate(result["context"], 1):
        print(f"  [{i}] {d.metadata['article_title'][:70]}")
        print(f"      {d.metadata['article_url']}")
