"""ColBERT M最適化実験: M=10 vs M=20 の比較。

baseline (P0) + ColBERT M=10 (P5) + ColBERT M=20 (P6) の3パターンで評価。

使い方:
    uv run python src/rerank_colbert_m_experiment.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

import lancedb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sudachipy import Dictionary

from ragatouille import RAGPretrainedModel

from src.search.bm25_search import load_chunks, search_bm25, tokenize
from src.eval.evaluate import (
    K_VALUES,
    calculate_precision_at_k,
    calculate_reciprocal_rank,
    calculate_recall_at_k,
    load_config,
    load_test_queries,
)
from src.search.hybrid_search import rrf_fusion
from src.experiments.rerank_experiment import colbert_rerank
from src.search.search import search as vector_search

EMBED_CONFIG = "config/embedding_e5_small.yaml"
CHUNKS_PATH = "data/processed/chunks_256_ov0.jsonl"
TABLE_NAME = "chunks_256_ov0_e5_small"
RETRIEVE_K = 20
RRF_K = 60
FINAL_TOP_K = max(K_VALUES)

PATTERNS = [
    {"id": "P0", "label": "baseline (hybrid only)", "M": None},
    {"id": "P5", "label": "ColBERT M=10", "M": 10},
    {"id": "P6", "label": "ColBERT M=20", "M": 20},
]


def main():
    queries = load_test_queries("eval/test_queries.yaml")
    print(f"テストクエリ数: {len(queries)}")

    tokenizer = Dictionary().create()
    config = load_config(EMBED_CONFIG)
    model_name = config["model"]["name"]
    query_prefix = config["model"].get("query_prefix", "")

    print(f"Embedding モデルロード: {model_name}")
    embed_model = SentenceTransformer(
        model_name,
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )

    db = lancedb.connect("data/lance")
    table = db.open_table(TABLE_NAME)

    chunks = load_chunks(CHUNKS_PATH)
    tokenized = [tokenize(tokenizer, c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print(f"チャンク数: {len(chunks)}")

    print("ColBERT モデルロード中...")
    colbert_model = RAGPretrainedModel.from_pretrained("answerdotai/JaColBERTv2.5")
    print()

    all_results = []
    for p in PATTERNS:
        print(f"{'='*60}")
        print(f"{p['id']}: {p['label']}")
        print(f"{'='*60}")

        recalls_by_k = {k: [] for k in K_VALUES}
        precisions_by_k = {k: [] for k in K_VALUES}
        rrs = []
        total_time = 0.0

        for q in queries:
            if "relevant_articles" not in q:
                continue
            correct_ids = q["relevant_articles"]

            start = time.time()

            vec = vector_search(q["query"], embed_model, table, query_prefix, top_k=RETRIEVE_K)
            bm = search_bm25(q["query"], bm25, chunks, tokenizer, top_k=RETRIEVE_K)

            if p["M"] is None:
                results = rrf_fusion(vec, bm, k=RRF_K, top_k=FINAL_TOP_K)
            else:
                candidates = rrf_fusion(vec, bm, k=RRF_K, top_k=p["M"])
                results = colbert_rerank(q["query"], candidates, colbert_model, top_k=FINAL_TOP_K)

            elapsed = time.time() - start
            total_time += elapsed

            rrs.append(calculate_reciprocal_rank(results, correct_ids))
            for k in K_VALUES:
                recalls_by_k[k].append(calculate_recall_at_k(results, correct_ids, k=k))
                precisions_by_k[k].append(calculate_precision_at_k(results, correct_ids, k=k))

        n = len(rrs)
        summary = {
            "pattern_id": p["id"],
            "label": p["label"],
            "M": p["M"],
            "num_queries": n,
            "mrr": sum(rrs) / n if n else 0,
            "avg_search_time_ms": total_time / n * 1000 if n else 0,
        }
        for k in K_VALUES:
            summary[f"recall_at_{k}"] = sum(recalls_by_k[k]) / n if n else 0
            summary[f"precision_at_{k}"] = sum(precisions_by_k[k]) / n if n else 0

        print(f"  Recall@1: {summary['recall_at_1']:.3f}")
        print(f"  Recall@3: {summary['recall_at_3']:.3f}")
        print(f"  Recall@5: {summary['recall_at_5']:.3f}")
        print(f"  Recall@10: {summary['recall_at_10']:.3f}")
        print(f"  MRR:      {summary['mrr']:.3f}")
        print(f"  検索時間: {summary['avg_search_time_ms']:.1f}ms")
        all_results.append(summary)
        print()

    # サマリー
    print(f"{'='*90}")
    print("結果サマリー")
    print(f"{'='*90}")
    print(f"{'pattern':<25} {'R@1':<8} {'R@3':<8} {'R@5':<8} {'R@10':<8} {'P@5':<8} {'MRR':<8} {'時間':<10}")
    print("-" * 90)
    for r in all_results:
        print(
            f"{r['pattern_id']+' '+r['label']:<25} "
            f"{r['recall_at_1']:.3f}   "
            f"{r['recall_at_3']:.3f}   "
            f"{r['recall_at_5']:.3f}   "
            f"{r['recall_at_10']:.3f}   "
            f"{r['precision_at_5']:.3f}   "
            f"{r['mrr']:.3f}   "
            f"{r['avg_search_time_ms']:.1f}ms"
        )

    # JSON 保存
    Path("benchmarks").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmarks/colbert_m_experiment_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "embed_config": EMBED_CONFIG,
            "chunks_path": CHUNKS_PATH,
            "table_name": TABLE_NAME,
            "retrieve_k": RETRIEVE_K,
            "rrf_k": RRF_K,
            "final_top_k": FINAL_TOP_K,
            "patterns": PATTERNS,
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {filename}")


if __name__ == "__main__":
    main()
