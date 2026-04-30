"""HyDE効果測定実験: プリコンピュート済み仮回答を使って検索精度を比較する。

パターン:
  P0: baseline（hybrid + ColBERT rerank）← 現在の最良
  P1: HyDE vector のみ（BM25なし、rerankなし）
  P2: HyDE + BM25 hybrid（rerankなし）
  P3: HyDE + BM25 hybrid + ColBERT rerank

使い方:
    uv run python src/hyde_experiment.py --hyde-answers data/processed/hyde_answers_qwen7b.jsonl
    uv run python src/hyde_experiment.py --hyde-answers data/processed/hyde_answers_qwen3b.jsonl
"""

import argparse
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
COLBERT_M = 10
FINAL_TOP_K = max(K_VALUES)


def hyde_vector_search(hypothetical_answer, model, table, passage_prefix, top_k):
    """HyDEベクトル検索: 仮回答をpassage_prefix付きでベクトル化して検索。"""
    text = passage_prefix + hypothetical_answer
    query_vector = model.encode([text])[0]
    return table.search(query_vector).distance_type("cosine").limit(top_k).to_list()


def load_hyde_answers(path):
    """プリコンピュート済み仮回答をロード。query -> hypothetical_answer のマップを返す。"""
    answers = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            answers[r["query"]] = r["hypothetical_answer"]
    return answers


def run_experiment(patterns, queries, hyde_answers, embed_model, table,
                   query_prefix, passage_prefix, bm25, chunks, tokenizer, colbert_model):
    """全パターンを実行して結果を返す。"""
    all_results = []

    for p in patterns:
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
            query = q["query"]

            start = time.time()

            if p["mode"] == "baseline":
                # P0: 通常 hybrid + ColBERT
                vec = vector_search(query, embed_model, table, query_prefix, top_k=RETRIEVE_K)
                bm = search_bm25(query, bm25, chunks, tokenizer, top_k=RETRIEVE_K)
                candidates = rrf_fusion(vec, bm, k=RRF_K, top_k=COLBERT_M)
                results = colbert_rerank(query, candidates, colbert_model, top_k=FINAL_TOP_K)

            elif p["mode"] == "hyde_vector_only":
                # P1: HyDE vector のみ
                hyp = hyde_answers[query]
                results = hyde_vector_search(hyp, embed_model, table, passage_prefix, top_k=FINAL_TOP_K)

            elif p["mode"] == "hyde_hybrid":
                # P2: HyDE vector + BM25 hybrid（rerankなし）
                hyp = hyde_answers[query]
                vec = hyde_vector_search(hyp, embed_model, table, passage_prefix, top_k=RETRIEVE_K)
                bm = search_bm25(query, bm25, chunks, tokenizer, top_k=RETRIEVE_K)
                results = rrf_fusion(vec, bm, k=RRF_K, top_k=FINAL_TOP_K)

            elif p["mode"] == "hyde_hybrid_colbert":
                # P3: HyDE vector + BM25 hybrid + ColBERT rerank
                hyp = hyde_answers[query]
                vec = hyde_vector_search(hyp, embed_model, table, passage_prefix, top_k=RETRIEVE_K)
                bm = search_bm25(query, bm25, chunks, tokenizer, top_k=RETRIEVE_K)
                candidates = rrf_fusion(vec, bm, k=RRF_K, top_k=COLBERT_M)
                results = colbert_rerank(query, candidates, colbert_model, top_k=FINAL_TOP_K)

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
            "mode": p["mode"],
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
        print(f"  MRR:      {summary['mrr']:.3f}")
        print(f"  検索時間: {summary['avg_search_time_ms']:.1f}ms")
        all_results.append(summary)
        print()

    return all_results


def main():
    parser = argparse.ArgumentParser(description="HyDE効果測定実験")
    parser.add_argument("--hyde-answers", required=True, help="プリコンピュート済み仮回答JSONL")
    args = parser.parse_args()

    patterns = [
        {"id": "P0", "label": "baseline (hybrid+ColBERT)", "mode": "baseline"},
        {"id": "P1", "label": "HyDE vector only", "mode": "hyde_vector_only"},
        {"id": "P2", "label": "HyDE + BM25 hybrid", "mode": "hyde_hybrid"},
        {"id": "P3", "label": "HyDE + BM25 hybrid + ColBERT", "mode": "hyde_hybrid_colbert"},
    ]

    # 共通初期化
    queries = load_test_queries("eval/test_queries.yaml")
    hyde_answers = load_hyde_answers(args.hyde_answers)
    print(f"テストクエリ数: {len(queries)}")
    print(f"HyDE仮回答: {args.hyde_answers} ({len(hyde_answers)}件)")

    config = load_config(EMBED_CONFIG)
    model_name = config["model"]["name"]
    query_prefix = config["model"].get("query_prefix", "")
    passage_prefix = config["model"].get("passage_prefix", "")

    print(f"Embedding: {model_name}")
    embed_model = SentenceTransformer(
        model_name,
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )

    db = lancedb.connect("data/lance")
    table = db.open_table(TABLE_NAME)

    tokenizer = Dictionary().create()
    chunks = load_chunks(CHUNKS_PATH)
    tokenized = [tokenize(tokenizer, c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print(f"チャンク数: {len(chunks)}")

    print("ColBERT モデルロード中...")
    colbert_model = RAGPretrainedModel.from_pretrained("answerdotai/JaColBERTv2.5")
    print()

    # 実験実行
    all_results = run_experiment(
        patterns, queries, hyde_answers, embed_model, table,
        query_prefix, passage_prefix, bm25, chunks, tokenizer, colbert_model,
    )

    # サマリー
    print(f"{'='*90}")
    print("結果サマリー")
    print(f"{'='*90}")
    print(f"{'pattern':<35} {'R@1':<8} {'R@3':<8} {'R@5':<8} {'MRR':<8} {'時間':<10}")
    print("-" * 90)
    for r in all_results:
        print(
            f"{r['pattern_id']+' '+r['label']:<35} "
            f"{r['recall_at_1']:.3f}   "
            f"{r['recall_at_3']:.3f}   "
            f"{r['recall_at_5']:.3f}   "
            f"{r['mrr']:.3f}   "
            f"{r['avg_search_time_ms']:.1f}ms"
        )

    # JSON保存
    Path("benchmarks").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hyde_label = Path(args.hyde_answers).stem.replace("hyde_answers_", "")
    filename = f"benchmarks/hyde_experiment_{hyde_label}_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "hyde_answers": args.hyde_answers,
            "embed_config": EMBED_CONFIG,
            "chunks_path": CHUNKS_PATH,
            "table_name": TABLE_NAME,
            "retrieve_k": RETRIEVE_K,
            "rrf_k": RRF_K,
            "colbert_m": COLBERT_M,
            "final_top_k": FINAL_TOP_K,
            "patterns": patterns,
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {filename}")


if __name__ == "__main__":
    main()
