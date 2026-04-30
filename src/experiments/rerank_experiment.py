"""リランク実験: hybrid + cross-encoder rerank の効果を測る。

4パターン比較:
    P0: baseline (hybrid のみ、rerank なし) ← 再測定
    P1: hybrid top-10 → xsmall rerank → top-5
    P2: hybrid top-20 → xsmall rerank → top-5
    P3: hybrid top-10 → small  rerank → top-5

使い方:
    python src/rerank_experiment.py

結果は benchmarks/rerank_experiment_YYYYMMDD_HHMMSS.json に保存される。
"""

import json
import time
from datetime import datetime
from pathlib import Path

import lancedb
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
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
from src.rerank.rerank import rerank
from src.search.search import search as vector_search

# 固定設定（最良構成）
EMBED_CONFIG = "config/embedding_e5_small.yaml"
CHUNKS_PATH = "data/processed/chunks_256_ov0.jsonl"
TABLE_NAME = "chunks_256_ov0_e5_small"
RETRIEVE_K = 20
RRF_K = 60
FINAL_TOP_K = max(K_VALUES)  # 10（全 K 指標を出すため）

# 実験パターン
PATTERNS = [
    {"id": "P0", "label": "baseline (hybrid only)", "rerank_config": None, "M": None},
    {"id": "P1", "label": "xsmall M=10", "rerank_config": "config/rerank_xsmall.yaml", "M": 10},
    {"id": "P2", "label": "xsmall M=20", "rerank_config": "config/rerank_xsmall.yaml", "M": 20},
    {"id": "P3", "label": "small  M=10", "rerank_config": "config/rerank_small.yaml", "M": 10},
    {"id": "P4", "label": "ruri-small M=10", "rerank_config": "config/rerank_ruri_small.yaml", "M": 10},
    {"id": "P5", "label": "JaColBERTv2.5 M=10", "rerank_config": "config/rerank_jacolbert.yaml", "M": 10},
]


def colbert_rerank(query, candidates, model, top_k=5):
    """ColBERT (ragatouille) でリランクする。

    ragatouille の rerank() は (content, score, rank) のリストを返す。
    返却順を candidates の元データと突き合わせて、元の dict 構造を保ったまま返す。
    """
    docs = [c["text"] for c in candidates]
    results = model.rerank(query=query, documents=docs, k=top_k)

    # ragatouille は result_index で元の docs のインデックスを返す
    out = []
    for r in results:
        idx = r["result_index"]
        c_copy = dict(candidates[idx])
        c_copy["_rerank_score"] = float(r["score"])
        out.append(c_copy)
    return out


def hybrid_search(query, model, table, query_prefix, bm25, chunks, tokenizer, top_m):
    """ベクトル検索 + BM25 + RRF で top_m 返す。"""
    vec = vector_search(query, model, table, query_prefix, top_k=RETRIEVE_K)
    bm = search_bm25(query, bm25, chunks, tokenizer, top_k=RETRIEVE_K)
    return rrf_fusion(vec, bm, k=RRF_K, top_k=top_m)


def run_pattern(pattern, queries, embed_model, table, query_prefix, bm25, chunks, tokenizer):
    """1パターンで全クエリを評価する。"""
    # リランクモデル
    rerank_model = None
    rerank_type = "cross_encoder"  # default
    if pattern["rerank_config"]:
        rconf = load_config(pattern["rerank_config"])
        rerank_type = rconf["reranker"].get("type", "cross_encoder")
        print(f"  リランクモデルロード: {rconf['reranker']['name']} ({rerank_type})")
        if rerank_type == "colbert":
            rerank_model = RAGPretrainedModel.from_pretrained(rconf["reranker"]["name"])
        else:
            rerank_model = CrossEncoder(
                rconf["reranker"]["name"],
                max_length=rconf["reranker"].get("max_length", 512),
                trust_remote_code=rconf["reranker"].get("trust_remote_code", False),
            )

    recalls_by_k = {k: [] for k in K_VALUES}
    precisions_by_k = {k: [] for k in K_VALUES}
    rrs = []
    total_time = 0.0

    for q in queries:
        if "relevant_articles" not in q:
            continue
        correct_ids = q["relevant_articles"]

        start = time.time()

        if rerank_model is None:
            # P0: hybrid のみ、top FINAL_TOP_K 取得
            results = hybrid_search(
                q["query"], embed_model, table, query_prefix,
                bm25, chunks, tokenizer, top_m=FINAL_TOP_K,
            )
        else:
            # P1-P5: hybrid で top-M → rerank で top FINAL_TOP_K
            candidates = hybrid_search(
                q["query"], embed_model, table, query_prefix,
                bm25, chunks, tokenizer, top_m=pattern["M"],
            )
            if rerank_type == "colbert":
                results = colbert_rerank(q["query"], candidates, rerank_model, top_k=FINAL_TOP_K)
            else:
                results = rerank(q["query"], candidates, rerank_model, top_k=FINAL_TOP_K)

        elapsed = time.time() - start
        total_time += elapsed

        rrs.append(calculate_reciprocal_rank(results, correct_ids))
        for k in K_VALUES:
            recalls_by_k[k].append(calculate_recall_at_k(results, correct_ids, k=k))
            precisions_by_k[k].append(calculate_precision_at_k(results, correct_ids, k=k))

    n = len(rrs)
    summary = {
        "pattern_id": pattern["id"],
        "label": pattern["label"],
        "rerank_config": pattern["rerank_config"],
        "M": pattern["M"],
        "num_queries": n,
        "mrr": sum(rrs) / n if n else 0,
        "avg_search_time_ms": total_time / n * 1000 if n else 0,
    }
    for k in K_VALUES:
        summary[f"recall_at_{k}"] = sum(recalls_by_k[k]) / n if n else 0
        summary[f"precision_at_{k}"] = sum(precisions_by_k[k]) / n if n else 0

    return summary


def main():
    # 共通初期化
    queries = load_test_queries("eval/test_queries.yaml")
    print(f"テストクエリ数: {len(queries)}")

    tokenizer = Dictionary().create()
    print("SudachiPy 初期化完了")

    config = load_config(EMBED_CONFIG)
    model_name = config["model"]["name"]
    query_prefix = config["model"].get("query_prefix", "")

    print(f"Embedding モデルロード: {model_name}")
    embed_model = SentenceTransformer(
        model_name,
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )

    print("LanceDB接続中...")
    db = lancedb.connect("data/lance")
    table = db.open_table(TABLE_NAME)

    print("チャンク読み込み + BM25 構築中...")
    chunks = load_chunks(CHUNKS_PATH)
    tokenized = [tokenize(tokenizer, c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print(f"  チャンク数: {len(chunks)}")
    print()

    # 4パターン実行
    all_results = []
    for p in PATTERNS:
        print(f"{'='*60}")
        print(f"{p['id']}: {p['label']}")
        print(f"{'='*60}")
        result = run_pattern(
            p, queries, embed_model, table, query_prefix,
            bm25, chunks, tokenizer,
        )
        print(f"  Recall@1: {result['recall_at_1']:.3f}")
        print(f"  Recall@3: {result['recall_at_3']:.3f}")
        print(f"  Recall@5: {result['recall_at_5']:.3f}")
        print(f"  MRR:      {result['mrr']:.3f}")
        print(f"  検索時間: {result['avg_search_time_ms']:.1f}ms")
        all_results.append(result)
        print()

    # サマリー
    print(f"{'='*90}")
    print("結果サマリー")
    print(f"{'='*90}")
    print(f"{'pattern':<20}{'Recall@1':<12}{'Recall@3':<12}{'Recall@5':<12}{'Precision@5':<14}{'MRR':<10}{'時間':<10}")
    print("-" * 90)
    for r in all_results:
        print(
            f"{r['pattern_id']+' '+r['label'][:15]:<20}"
            f"{r['recall_at_1']:.3f}       "
            f"{r['recall_at_3']:.3f}       "
            f"{r['recall_at_5']:.3f}       "
            f"{r['precision_at_5']:.3f}         "
            f"{r['mrr']:.3f}     "
            f"{r['avg_search_time_ms']:.1f}ms"
        )

    # ベスト
    by_r1 = max(all_results, key=lambda x: x["recall_at_1"])
    by_mrr = max(all_results, key=lambda x: x["mrr"])
    print(f"\nベスト Recall@1: {by_r1['pattern_id']} ({by_r1['label']}) = {by_r1['recall_at_1']:.3f}")
    print(f"ベスト MRR:       {by_mrr['pattern_id']} ({by_mrr['label']}) = {by_mrr['mrr']:.3f}")

    # JSON 保存
    Path("benchmarks").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmarks/rerank_experiment_{timestamp}.json"
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
