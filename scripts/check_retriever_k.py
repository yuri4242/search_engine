"""LangChain Retriever の k 指定が効いているか確認。"""
import lancedb
import yaml

from langchain_community.vectorstores import LanceDB

from src.embeddings import E5Embeddings


def main():
    config = yaml.safe_load(open("config/langchain_rag.yaml"))

    emb = E5Embeddings(
        model_name=config["embedding"]["model"],
        encode_kwargs={"normalize_embeddings": config["embedding"]["normalize"]},
    )
    db = lancedb.connect(config["vectorstore"]["path"])
    vectorstore = LanceDB(
        connection=db,
        table_name=config["vectorstore"]["table_name"],
        embedding=emb,
    )

    query = "RAGのチャンク分割"

    for k in [4, 10, 20]:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        results = retriever.invoke(query)
        print(f"k={k} を指定 → 戻り値の件数: {len(results)}")


if __name__ == "__main__":
    main()
