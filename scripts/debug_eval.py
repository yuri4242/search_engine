"""evaluate_vector.py の中身を1クエリずつ詳細表示してデバッグする。

archive 側で同じ表示を出して、どこが違うか目視比較する。
"""
import yaml
import lancedb

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
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    with open("eval/test_queries.yaml", encoding="utf-8") as f:
        queries = yaml.safe_load(f)

    # 最初の3クエリで詳細確認
    for i, q in enumerate(queries[:3], 1):
        query = q["query"]
        relevant_raw = q["relevant_articles"]
        relevant = set(relevant_raw)
        results = vector_retriever.invoke(query)
        article_ids = [d.metadata["article_id"] for d in results]

        print(f"=== クエリ {i}: {query} ===")
        print(f"relevant_articles の型: {type(relevant_raw).__name__}")
        print(f"relevant_articles の中身: {relevant_raw}")
        print(f"正解（set化後）: {relevant}")
        print(f"取得件数: {len(article_ids)}")
        print(f"取得 article_ids（top 10）: {article_ids[:10]}")
        print(f"取得 article_ids の型サンプル: {type(article_ids[0]).__name__}")
        print(f"top-1 hit: {article_ids[0] in relevant}")
        print(f"top-5 hit: {any(aid in relevant for aid in article_ids[:5])}")
        print()


if __name__ == "__main__":
    main()
