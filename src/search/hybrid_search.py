"""ハイブリッド検索（ベクトル + BM25 を RRF で統合）。

使い方:
    python src/hybrid_search.py --config config/embedding_ruri_small.yaml --query "RAG"
    python src/hybrid_search.py --config config/embedding_ruri_small.yaml --query-file eval/test_queries.yaml
"""

import argparse
import json
import time

import lancedb
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sudachipy import Dictionary

from src.search.search import search as vector_search
from src.search.bm25_search import tokenize, search_bm25, load_chunks


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_test_queries(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def rrf_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    """
    RRF（Reciprocal Rank Fusion）で2つの検索結果を統合する。

    Args:
        vector_results: ベクトル検索の結果（上位から順、各要素にarticle_id, chunk_indexを含む）
        bm25_results: BM25検索の結果（上位から順、各要素にarticle_id, chunk_indexを含む）
        k: RRFのパラメータ（デフォルト60）
        top_k: 最終的に返す件数

    Returns:
        RRFスコアで降順ソートされた上位top_k件のチャンク辞書のリスト。
        各要素に "_rrf_score" が追加される。
    """
    # RRF スコア計算式:
    #   RRF_score(doc) = Σ 1 / (k + rank(doc))
    #   （各検索結果でのdocの順位rankの逆数を足し合わせる）
    #
    # 1. 空の辞書 scores を用意（key=チャンク識別子, value=RRFスコア）
    scores = {}
    # 2. 空の辞書 chunks_by_key を用意（最終的にチャンク本体を返すため）
    chunks_by_key = {}
    # 3. vector_results をループして、各チャンクのRRFスコアを加算
    #    - 順位 rank は enumerate で取れる（0始まりなのでrank + 1にする）
    #    - チャンクの識別子 key は (article_id, chunk_index) のタプルが使える
    #    - scores[key] += 1 / (k + rank)
    for rank, chunk in enumerate(vector_results):
        key = (chunk["article_id"], chunk["chunk_index"])
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        chunks_by_key[key] = chunk
    # 4. bm25_results にも同じ処理をする
    for rank, chunk in enumerate(bm25_results):
        key = (chunk["article_id"], chunk["chunk_index"])
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        chunks_by_key[key] = chunk
    # 5. scores を値（RRFスコア）で降順ソートして上位top_k件取り出す
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    results = []
    for key in sorted_keys[:top_k]:
        chunk = dict(chunks_by_key[key])
        chunk["_rrf_score"] = scores[key]
        results.append(chunk)
    # 6. 各チャンクに "_rrf_score" を追加して返す
    return results


def print_results(query: str, results: list[dict]) -> None:
    """検索結果を表示する。"""
    print(f"\n[クエリ] {query}")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        rrf = r.get("_rrf_score", 0)
        print(f"  {i}. ({r['article_id'][:8]}... chunk {r['chunk_index']:2d}) "
              f"RRF={rrf:.4f}")
        print(f"     {r['article_title'][:70]}")
        print(f"     本文: {r['text'][:80]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="ベクトル + BM25 のハイブリッド検索")
    parser.add_argument("--config", required=True, help="Embedding設定ファイル")
    parser.add_argument("--query", help="単発クエリ文字列")
    parser.add_argument("--query-file", help="テストクエリYAMLファイル")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieve-k", type=int, default=20,
                        help="RRFに渡す前の各検索の取得件数（デフォルト20）")
    parser.add_argument("--rrf-k", type=int, default=60,
                        help="RRFのkパラメータ（デフォルト60）")
    args = parser.parse_args()

    if not args.query and not args.query_file:
        parser.error("--query か --query-file のどちらかを指定してください")

    # 設定読み込み
    config = load_config(args.config)
    model_name = config["model"]["name"]
    query_prefix = config["model"].get("query_prefix", "")
    trust_remote_code = config["model"].get("trust_remote_code", False)
    db_path = config["lancedb"]["path"]
    table_name = config["lancedb"]["table_name"]

    print(f"設定: {args.config}")
    print(f"  モデル: {model_name}")
    print(f"  取得件数（各検索）: {args.retrieve_k}")
    print(f"  RRF k: {args.rrf_k}")
    print(f"  最終top-k: {args.top_k}")

    # チャンク読み込み（BM25用）
    print("\nチャンク読み込み中...")
    chunks = load_chunks(args.chunks)

    # Embeddingモデルロード
    print(f"\nEmbeddingモデルをロード中...")
    start = time.time()
    model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
    print(f"  ロード完了（{time.time() - start:.1f}秒）")

    # LanceDBテーブル
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)

    # SudachiPy初期化
    print("SudachiPy 初期化中...")
    tokenizer = Dictionary().create()

    # BM25インデックス構築
    print("BM25インデックス構築中...")
    start = time.time()
    tokenized_docs = [tokenize(tokenizer, c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_docs)
    print(f"  構築完了（{time.time() - start:.1f}秒）")

    def run_one_query(query: str):
        vec_results = vector_search(query, model, table, query_prefix, top_k=args.retrieve_k)
        bm25_results = search_bm25(query, bm25, chunks, tokenizer, top_k=args.retrieve_k)
        return rrf_fusion(vec_results, bm25_results, k=args.rrf_k, top_k=args.top_k)

    # 検索実行
    if args.query:
        results = run_one_query(args.query)
        print_results(args.query, results)
    else:
        queries = load_test_queries(args.query_file)
        print(f"\nテストクエリ数: {len(queries)}")
        for q in queries:
            results = run_one_query(q["query"])
            print_results(q["query"], results)
            if "relevant_articles" in q:
                correct_ids = [a[:8] for a in q["relevant_articles"]]
                returned_ids = [r["article_id"][:8] for r in results]
                hit = any(cid in returned_ids for cid in correct_ids)
                print(f"     正解記事: {correct_ids} / ヒット: {'✓' if hit else '✗'}")


if __name__ == "__main__":
    main()
