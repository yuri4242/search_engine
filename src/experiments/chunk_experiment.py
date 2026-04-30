"""チャンクサイズ実験: 2モデル × 4チャンクサイズの全8パターンを一括評価する。

使い方:
    python src/chunk_experiment.py

結果は benchmarks/chunk_experiment_YYYYMMDD_HHMMSS.json に保存される。
"""

import json
import time
from datetime import datetime
from pathlib import Path

import lancedb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sudachipy import Dictionary

from src.chunk.build_chunks import build_chunks
from src.embed.embed_chunks import embed_chunks, save_to_lancedb, load_chunks, load_config
from src.search.bm25_search import tokenize, search_bm25
from src.search.search import search as vector_search
from src.search.hybrid_search import rrf_fusion
from src.eval.evaluate import (
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    load_test_queries,
)

# 実験条件
CHUNK_SIZES = [128, 256, 384, 512]
MODEL_CONFIGS = [
    "config/embedding_ruri_base.yaml",
    "config/embedding_e5_small.yaml",
]

# 評価パラメータ
TOP_K = 5
RETRIEVE_K = 20
RRF_K = 60


def short_model_name(config_path: str) -> str:
    """config ファイル名から短縮モデル名を取り出す。"""
    # config/embedding_ruri_base.yaml → ruri_base
    name = Path(config_path).stem.replace("embedding_", "")
    return name


def run_hybrid_evaluation(
    queries: list[dict],
    model: SentenceTransformer,
    table,
    query_prefix: str,
    bm25: BM25Okapi,
    chunks: list[dict],
    tokenizer,
) -> dict:
    """全クエリで hybrid 検索を実行して Recall@k と MRR を集計する。"""
    recalls = []
    rrs = []
    total_time = 0.0

    for q in queries:
        if "relevant_articles" not in q:
            continue

        start = time.time()
        vec_results = vector_search(q["query"], model, table, query_prefix, top_k=RETRIEVE_K)
        bm25_results = search_bm25(q["query"], bm25, chunks, tokenizer, top_k=RETRIEVE_K)
        results = rrf_fusion(vec_results, bm25_results, k=RRF_K, top_k=TOP_K)
        total_time += time.time() - start

        recalls.append(calculate_recall_at_k(results, q["relevant_articles"], k=TOP_K))
        rrs.append(calculate_reciprocal_rank(results, q["relevant_articles"]))

    return {
        "recall_at_k": sum(recalls) / len(recalls) if recalls else 0,
        "mrr": sum(rrs) / len(rrs) if rrs else 0,
        "avg_search_time_ms": total_time / len(recalls) * 1000 if recalls else 0,
        "num_queries": len(recalls),
    }


def main():
    # テストクエリ読み込み
    queries = load_test_queries("eval/test_queries.yaml")
    print(f"テストクエリ数: {len(queries)}")

    # SudachiPy 初期化（1回だけ）
    tokenizer = Dictionary().create()
    print("SudachiPy 初期化完了")
    print()

    # 全実験結果を保存
    all_results = []

    for chunk_size in CHUNK_SIZES:
        print(f"{'='*60}")
        print(f"チャンクサイズ: {chunk_size}")
        print(f"{'='*60}")

        # 1. チャンク分割
        chunks_path = f"data/processed/chunks_{chunk_size}.jsonl"
        print(f"チャンク分割中 → {chunks_path}")
        build_chunks(output_path=chunks_path, chunk_size=chunk_size)
        chunks = load_chunks(chunks_path)
        print(f"  チャンク数: {len(chunks)}")

        # 2. BM25 インデックス構築（このチャンクサイズ専用）
        print("BM25 インデックス構築中...")
        start = time.time()
        tokenized = [tokenize(tokenizer, c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        print(f"  完了（{time.time() - start:.1f}秒）")

        # 3. 各モデルでベクトル化 + 評価
        for config_path in MODEL_CONFIGS:
            config = load_config(config_path)
            model_name = config["model"]["name"]
            short_name = short_model_name(config_path)
            table_name = f"chunks_{chunk_size}_{short_name}"

            print(f"\n--- モデル: {model_name} ---")

            # ベクトル化
            print("モデルロード + ベクトル化中...")
            start = time.time()
            model = SentenceTransformer(
                model_name,
                trust_remote_code=config["model"].get("trust_remote_code", False),
            )
            chunks_copy = [dict(c) for c in chunks]  # 元を汚染しないコピー
            chunks_with_vec = embed_chunks(
                chunks_copy, model, config["model"].get("passage_prefix", "")
            )
            save_to_lancedb(chunks_with_vec, "data/lance", table_name)
            vec_time = time.time() - start
            print(f"  完了（{vec_time:.1f}秒）")

            # 評価
            print("評価中...")
            db = lancedb.connect("data/lance")
            table = db.open_table(table_name)
            result = run_hybrid_evaluation(
                queries, model, table,
                config["model"].get("query_prefix", ""),
                bm25, chunks, tokenizer,
            )
            print(f"  Recall@{TOP_K}: {result['recall_at_k']:.3f}")
            print(f"  MRR:       {result['mrr']:.3f}")
            print(f"  平均検索時間: {result['avg_search_time_ms']:.1f}ms")

            all_results.append({
                "chunk_size": chunk_size,
                "model": model_name,
                "table_name": table_name,
                "vectorize_time_sec": vec_time,
                **result,
            })

        print()

    # サマリー表示
    print(f"\n{'='*80}")
    print("実験結果サマリー")
    print(f"{'='*80}")
    print(f"{'chunk_size':<12}{'model':<35}{'Recall@5':<12}{'MRR':<10}{'検索時間':<12}")
    print("-" * 80)
    for r in all_results:
        print(
            f"{r['chunk_size']:<12}"
            f"{r['model']:<35}"
            f"{r['recall_at_k']:.3f}       "
            f"{r['mrr']:.3f}     "
            f"{r['avg_search_time_ms']:.1f}ms"
        )

    # ベストを表示
    by_recall = max(all_results, key=lambda x: (x["recall_at_k"], x["mrr"]))
    by_mrr = max(all_results, key=lambda x: (x["mrr"], x["recall_at_k"]))
    print(f"\nベスト Recall@5: chunk_size={by_recall['chunk_size']}, "
          f"model={by_recall['model']}, Recall@5={by_recall['recall_at_k']:.3f}")
    print(f"ベスト MRR:       chunk_size={by_mrr['chunk_size']}, "
          f"model={by_mrr['model']}, MRR={by_mrr['mrr']:.3f}")

    # JSON保存
    Path("benchmarks").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmarks/chunk_experiment_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "chunk_sizes": CHUNK_SIZES,
            "models": MODEL_CONFIGS,
            "top_k": TOP_K,
            "retrieve_k": RETRIEVE_K,
            "rrf_k": RRF_K,
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {filename}")


if __name__ == "__main__":
    main()
