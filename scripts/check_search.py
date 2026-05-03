"""LangChain Retriever 経由 vs LanceDB 直接呼び出しで検索結果が違うか確認。

Vector retriever のスコアが旧と一致しない原因（distance_type 等）を切り分ける。
"""
import lancedb

from langchain_community.vectorstores import LanceDB

from src.embeddings import E5Embeddings


def print_results(label, results, get_meta):
    print(f"=== {label} ===")
    for i, r in enumerate(results, 1):
        meta = get_meta(r)
        print(f"{i:2d}. {meta['article_id']} - {meta['article_title'][:40]}")
    print()


def main():
    emb = E5Embeddings(
        model_name="intfloat/multilingual-e5-small",
        encode_kwargs={"normalize_embeddings": True},
    )

    db = lancedb.connect("data/lance")
    table = db.open_table("chunks_256_ov0_e5_small_lc")

    query = "RAGのチャンク分割"
    qvec = emb.embed_query(query)

    # 1. LanceDB 直接（cosine 明示）
    results = table.search(qvec).distance_type("cosine").limit(10).to_list()
    print_results("LanceDB 直接（cosine 明示）", results, lambda r: r["metadata"])

    # 2. LanceDB 直接（distance_type 未指定）
    results = table.search(qvec).limit(10).to_list()
    print_results("LanceDB 直接（デフォルト distance_type）", results, lambda r: r["metadata"])

    # 3. LangChain Retriever 経由
    vectorstore = LanceDB(
        connection=db,
        table_name="chunks_256_ov0_e5_small_lc",
        embedding=emb,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(query)
    print_results("LangChain Retriever 経由", docs, lambda d: d.metadata)


if __name__ == "__main__":
    main()
