"""BM25でチャンクを検索する。

使い方:
    python src/bm25_search.py --query "RAGのチャンク分割"
    python src/bm25_search.py --query-file eval/test_queries.yaml
"""

import argparse
import json
import time

import yaml
from rank_bm25 import BM25Okapi
from sudachipy import Dictionary, SplitMode


def load_chunks(chunks_path: str) -> list[dict]:
    """JSONLからチャンクを読む（embed_chunks.pyと同じ）。"""
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def load_test_queries(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def tokenize(tokenizer, text: str) -> list[str]:
    """
    日本語テキストをSudachiPyでトークナイズして表層形のリストを返す。

    Args:
        tokenizer: SudachiPy の tokenizer (Dictionary().create())
        text: 日本語テキスト

    Returns:
        単語のリスト 例: ["RAG", "の", "チャンク", "分割", ...]
    """
    morphemes = tokenizer.tokenize(text, mode=SplitMode.C)
    return [m.normalized_form() for m in morphemes]


def search_bm25(
    query: str,
    bm25: BM25Okapi,
    chunks: list[dict],
    tokenizer,
    top_k: int = 5,
) -> list[dict]:
    """
    BM25で上位top_k件を返す。

    Args:
        query: 検索クエリ
        bm25: 構築済みのBM25Okapiインデックス
        chunks: 元のチャンクリスト（bm25のインデックスと同じ順序）
        tokenizer: SudachiPyのtokenizer
        top_k: 上位何件を返すか

    Returns:
        上位top_k件のチャンク辞書のリスト。各チャンクに "_score" が追加される。
    """
    import numpy as np

    query_tokens = tokenize(tokenizer, query)
    scores = bm25.get_scores(query_tokens)
    top_idx = np.argsort(-scores)[:top_k]

    results = []
    for i in top_idx:
        chunk = dict(chunks[i])
        chunk["_score"] = float(scores[i])
        results.append(chunk)
    return results


def print_results(query: str, results: list[dict]) -> None:
    """検索結果を表示する（search.pyと同じ見た目）。"""
    print(f"\n[クエリ] {query}")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        score = r.get("_score", 0)
        print(f"  {i}. ({r['article_id'][:8]}... chunk {r['chunk_index']:2d}) "
              f"score={score:.3f}")
        print(f"     {r['article_title'][:70]}")
        print(f"     本文: {r['text'][:80]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="BM25でチャンクを検索する")
    parser.add_argument("--query", help="単発クエリ文字列")
    parser.add_argument("--query-file", help="テストクエリYAMLファイル")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not args.query and not args.query_file:
        parser.error("--query か --query-file のどちらかを指定してください")

    # チャンク読み込み
    print("チャンク読み込み中...")
    chunks = load_chunks(args.chunks)
    print(f"  チャンク数: {len(chunks)}")

    # SudachiPy の tokenizer 準備
    print("\nSudachiPy 初期化中...")
    tokenizer = Dictionary().create()

    # 全チャンクをトークナイズ
    print("\n全チャンクをトークナイズ中...")
    start = time.time()
    tokenized_docs = [tokenize(tokenizer, c["text"]) for c in chunks]
    print(f"  完了（{time.time() - start:.1f}秒）")

    # BM25インデックス構築
    print("\nBM25インデックス構築中...")
    start = time.time()
    bm25 = BM25Okapi(tokenized_docs)
    print(f"  完了（{time.time() - start:.1f}秒）")

    # 検索実行
    if args.query:
        results = search_bm25(args.query, bm25, chunks, tokenizer, args.top_k)
        print_results(args.query, results)
    else:
        queries = load_test_queries(args.query_file)
        print(f"\nテストクエリ数: {len(queries)}")
        for q in queries:
            results = search_bm25(q["query"], bm25, chunks, tokenizer, args.top_k)
            print_results(q["query"], results)
            if "relevant_articles" in q:
                correct_ids = [a[:8] for a in q["relevant_articles"]]
                returned_ids = [r["article_id"][:8] for r in results]
                hit = any(cid in returned_ids for cid in correct_ids)
                print(f"     正解記事: {correct_ids} / ヒット: {'✓' if hit else '✗'}")


if __name__ == "__main__":
    main()
