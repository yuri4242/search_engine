"""archive用フラットテーブルと LC用 metadata JSON テーブルで、同じクエリの検索結果を比較。

両テーブルとも同じ chunks_256_ov0.jsonl から E5Embeddings で作ったはずだが、
結果が違うなら build_index.py か embed_chunks.py のどちらかでベクトル生成が違っている。
"""
import lancedb

from src.embeddings import E5Embeddings


def main():
    emb = E5Embeddings(
        model_name="intfloat/multilingual-e5-small",
        encode_kwargs={"normalize_embeddings": True},
    )

    db = lancedb.connect("data/lance")

    query = "RAGのチャンク分割"
    qvec = emb.embed_query(query)

    # archive 用フラット形式
    flat = db.open_table("chunks_256_ov0_e5_small")
    print(f"=== archive 用テーブル（フラット）件数: {flat.count_rows()} ===")
    results = flat.search(qvec).distance_type("cosine").limit(10).to_list()
    for i, r in enumerate(results, 1):
        print(f"{i:2d}. {r['article_id']} - {r['article_title'][:40]}")

    print()

    # LC 用 metadata JSON 形式
    lc = db.open_table("chunks_256_ov0_e5_small_lc")
    print(f"=== LC 用テーブル（metadata JSON）件数: {lc.count_rows()} ===")
    results = lc.search(qvec).distance_type("cosine").limit(10).to_list()
    for i, r in enumerate(results, 1):
        print(f"{i:2d}. {r['metadata']['article_id']} - {r['metadata']['article_title'][:40]}")


if __name__ == "__main__":
    main()
