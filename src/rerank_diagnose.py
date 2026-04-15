"""リランクで Recall@3 が下がったクエリを特定し、top-10 の中身を比較する。

仮説検証用: 「評価軸のズレ」（リランカーは意味的に関連するチャンクを上位に押し上げているが、
記事単位の正解ラベルと噛み合わない）が本当に起きているかを目視で確認する。
"""

import lancedb
from sentence_transformers import CrossEncoder, SentenceTransformer
from sudachipy import Dictionary
from rank_bm25 import BM25Okapi

from bm25_search import load_chunks, search_bm25, tokenize
from evaluate import load_config, load_test_queries, calculate_recall_at_k
from hybrid_search import rrf_fusion
from rerank import rerank
from search import search as vector_search

EMBED_CONFIG = "config/embedding_e5_small.yaml"
RERANK_MODEL = "hotchpotch/japanese-reranker-cross-encoder-xsmall-v1"
CHUNKS_PATH = "data/processed/chunks_256_ov0.jsonl"
TABLE_NAME = "chunks_256_ov0_e5_small"
RETRIEVE_K = 20
RRF_K = 60
M = 10  # rerank候補数
TOP_K = 10  # 表示と評価で使う


def main():
    queries = load_test_queries("eval/test_queries.yaml")
    tokenizer = Dictionary().create()

    config = load_config(EMBED_CONFIG)
    embed_model = SentenceTransformer(
        config["model"]["name"],
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )
    query_prefix = config["model"].get("query_prefix", "")

    db = lancedb.connect("data/lance")
    table = db.open_table(TABLE_NAME)

    chunks = load_chunks(CHUNKS_PATH)
    tokenized = [tokenize(tokenizer, c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    rerank_model = CrossEncoder(RERANK_MODEL, max_length=512)

    # 全クエリで baseline と rerank の Recall@3 を比較
    diverged = []
    for q in queries:
        if "relevant_articles" not in q:
            continue
        correct_ids = q["relevant_articles"]

        vec = vector_search(q["query"], embed_model, table, query_prefix, top_k=RETRIEVE_K)
        bm = search_bm25(q["query"], bm25, chunks, tokenizer, top_k=RETRIEVE_K)
        baseline = rrf_fusion(vec, bm, k=RRF_K, top_k=TOP_K)

        candidates = rrf_fusion(vec, bm, k=RRF_K, top_k=M)
        reranked = rerank(q["query"], candidates, rerank_model, top_k=TOP_K)

        b_r3 = calculate_recall_at_k(baseline, correct_ids, k=3)
        r_r3 = calculate_recall_at_k(reranked, correct_ids, k=3)

        if r_r3 < b_r3:
            diverged.append((q, correct_ids, baseline, reranked, b_r3, r_r3))

    print(f"=== Recall@3 が悪化したクエリ: {len(diverged)}件 ===\n")

    for q, correct_ids, baseline, reranked, b_r3, r_r3 in diverged:
        print("=" * 90)
        print(f"クエリ: {q['query']}")
        print(f"正解 article_id: {correct_ids}")
        print(f"baseline Recall@3 = {b_r3:.2f} → rerank Recall@3 = {r_r3:.2f}")
        print()
        print("--- baseline top-10 ---")
        for i, r in enumerate(baseline, 1):
            mark = "✅" if r["article_id"] in correct_ids else "  "
            print(f"  {i:2}. {mark} [{r['article_id'][:10]}] c{r.get('chunk_index', '?'):>2} {r['text'][:60]}")
        print()
        print("--- rerank top-10 ---")
        for i, r in enumerate(reranked, 1):
            mark = "✅" if r["article_id"] in correct_ids else "  "
            score = r.get("_rerank_score", 0)
            print(f"  {i:2}. {mark} [{r['article_id'][:10]}] c{r.get('chunk_index', '?'):>2} score={score:+.3f} {r['text'][:50]}")
        print()


if __name__ == "__main__":
    main()
