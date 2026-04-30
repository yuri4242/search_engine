"""検索精度を Recall@k と MRR で評価する。

使い方:
    python src/evaluate.py --mode vector --config config/embedding_ruri_small.yaml
    python src/evaluate.py --mode bm25
    python src/evaluate.py --mode hybrid --config config/embedding_ruri_small.yaml
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import lancedb
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sudachipy import Dictionary

from src.search.search import search as vector_search
from src.search.bm25_search import tokenize, search_bm25, load_chunks
from src.search.hybrid_search import rrf_fusion


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_test_queries(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


K_VALUES = [1, 3, 5, 10]


def calculate_precision_at_k(
    results: list[dict],
    correct_article_ids: list[str],
    k: int,
) -> float:
    """Precision@k: top-K チャンクのうち正解記事に属するチャンクの割合。

    例: top-5 = [A_c3, A_c5, B_c1, C_c0, D_c2], correct=[A]
        → hits=2, precision=2/5=0.4
    「LLM に渡す K チャンクのうち、何割が正解記事由来か」を測る。
    """
    top_k = results[:k]
    hits = sum(1 for r in top_k if r["article_id"] in correct_article_ids)
    return hits / k if k > 0 else 0.0


def calculate_recall_at_k(
    results: list[dict],
    correct_article_ids: list[str],
    k: int = 5,
) -> float:
    """
    1クエリに対する Recall@k を計算する。

    Args:
        results: 検索結果のリスト（上位から順、各要素に "article_id" を含む）
        correct_article_ids: 正解記事IDのリスト（relevant_articles）
        k: 上位何件を見るか

    Returns:
        Recall@k の値（0.0 〜 1.0）
        = 正解記事のうち、上位k件に含まれていた件数 / 正解記事の総数

    例:
        正解2件、上位5件に1件だけヒット → 1/2 = 0.5
        正解1件、上位5件にヒット          → 1/1 = 1.0
        正解1件、上位5件にヒットなし      → 0/1 = 0.0
    """
    # 1. results の上位k件から article_id のセットを作る
    top_k_ids = set(r["article_id"] for r in results[:k])
    # 2. correct_article_ids の中で、上位k件に含まれる数をカウント
    hits = sum(1 for cid in correct_article_ids if cid in top_k_ids)
    # 3. カウント / 正解記事の総数 を返す
    #    （正解が0件のときは ZeroDivision 防止で 0.0 を返す）
    return hits / len(correct_article_ids) if correct_article_ids else 0.0


def calculate_reciprocal_rank(
    results: list[dict],
    correct_article_ids: list[str],
) -> float:
    """
    1クエリに対する Reciprocal Rank を計算する（MRRの平均前の値）。

    Args:
        results: 検索結果のリスト（上位から順）
        correct_article_ids: 正解記事IDのリスト

    Returns:
        Reciprocal Rank の値
        = 正解記事のうち最も上位にあった順位の逆数
        = 正解が1位なら 1.0、2位なら 0.5、3位なら 0.333...
        = 結果に含まれない場合は 0.0

    例:
        正解[A, B]、検索結果 [X, A, B, Y] → A が2位 → 1/2 = 0.5
        正解[A]、検索結果 [X, Y, Z, A]     → A が4位 → 1/4 = 0.25
        正解[A]、検索結果 [X, Y, Z, W]     → 0.0

    注意: MRR（Mean Reciprocal Rank）は複数クエリの RR の平均。
          この関数は1クエリ分の RR を返す。平均は呼び出し側で計算する。
    """
    # 1. results を上から順に見て、article_id が correct_article_ids に含まれるかチェック
    # 2. 最初に見つかった位置（1-indexed）の逆数を返す
    # 3. 見つからなかったら 0.0 を返す
    for rank, r in enumerate(results, start=1):
        if r["article_id"] in correct_article_ids:
            return 1.0 / rank
    return 0.0


def run_search(mode: str, query: str, resources: dict) -> list[dict]:
    """指定モードで検索を実行する。resourcesには初期化済みのオブジェクトが入っている。"""
    if mode == "vector":
        return vector_search(
            query, resources["model"], resources["table"],
            resources["query_prefix"], top_k=resources["top_k"]
        )
    elif mode == "bm25":
        return search_bm25(
            query, resources["bm25"], resources["chunks"],
            resources["tokenizer"], top_k=resources["top_k"]
        )
    elif mode == "hybrid":
        vec = vector_search(
            query, resources["model"], resources["table"],
            resources["query_prefix"], top_k=resources["retrieve_k"]
        )
        bm = search_bm25(
            query, resources["bm25"], resources["chunks"],
            resources["tokenizer"], top_k=resources["retrieve_k"]
        )
        return rrf_fusion(vec, bm, k=resources["rrf_k"], top_k=resources["top_k"])
    else:
        raise ValueError(f"Unknown mode: {mode}")


def evaluate(mode: str, queries: list[dict], resources: dict) -> dict:
    """全クエリを実行して Recall@K / Precision@K / MRR を集計する。

    K_VALUES (= [1, 3, 5, 10]) のすべてで Recall@K と Precision@K を計算する。
    検索は max(K_VALUES) 件返し、各 K で切り出して評価する。
    """
    recalls_by_k = {k: [] for k in K_VALUES}
    precisions_by_k = {k: [] for k in K_VALUES}
    rrs = []
    per_query = []
    total_time = 0.0

    for q in queries:
        if "relevant_articles" not in q:
            continue  # 正解なしのクエリはスキップ
        query_text = q["query"]
        correct_ids = q["relevant_articles"]

        start = time.time()
        results = run_search(mode, query_text, resources)
        elapsed = time.time() - start
        total_time += elapsed

        rr = calculate_reciprocal_rank(results, correct_ids)
        rrs.append(rr)

        q_metrics = {}
        for k in K_VALUES:
            r = calculate_recall_at_k(results, correct_ids, k=k)
            p = calculate_precision_at_k(results, correct_ids, k=k)
            recalls_by_k[k].append(r)
            precisions_by_k[k].append(p)
            q_metrics[f"recall_at_{k}"] = r
            q_metrics[f"precision_at_{k}"] = p

        per_query.append({
            "query": query_text,
            "correct_ids": correct_ids,
            "returned_ids": [r["article_id"] for r in results],
            **q_metrics,
            "reciprocal_rank": rr,
            "search_time_sec": elapsed,
        })

    n = len(rrs)
    summary = {
        "mode": mode,
        "num_queries": n,
        "mrr": sum(rrs) / n if n else 0,
        "avg_search_time_ms": total_time / n * 1000 if n else 0,
        "per_query": per_query,
    }
    for k in K_VALUES:
        summary[f"recall_at_{k}"] = sum(recalls_by_k[k]) / n if n else 0
        summary[f"precision_at_{k}"] = sum(precisions_by_k[k]) / n if n else 0

    return summary


def save_result(result: dict, mode: str, model_name: str | None) -> Path:
    """結果をbenchmarks/にJSON保存する。"""
    Path("benchmarks").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_name:
        safe_name = model_name.replace("/", "_")
        filename = f"benchmarks/{timestamp}_{mode}_{safe_name}.json"
    else:
        filename = f"benchmarks/{timestamp}_{mode}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return Path(filename)


def main():
    parser = argparse.ArgumentParser(
        description="検索精度を Recall@K / Precision@K / MRR で評価（K=1,3,5,10）"
    )
    parser.add_argument("--mode", required=True, choices=["vector", "bm25", "hybrid"])
    parser.add_argument("--config", help="Embedding設定ファイル（vector, hybridで必要）")
    parser.add_argument("--query-file", default="eval/test_queries.yaml")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--retrieve-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()

    if args.mode in ("vector", "hybrid") and not args.config:
        parser.error(f"--mode {args.mode} には --config が必要")

    # 初期化: 検索は max(K_VALUES) 件返す必要がある
    resources = {
        "top_k": max(K_VALUES),
        "retrieve_k": args.retrieve_k,
        "rrf_k": args.rrf_k,
    }

    model_name = None
    if args.mode in ("vector", "hybrid"):
        config = load_config(args.config)
        model_name = config["model"]["name"]
        print(f"モデルロード: {model_name}")
        resources["model"] = SentenceTransformer(
            model_name,
            trust_remote_code=config["model"].get("trust_remote_code", False)
        )
        resources["query_prefix"] = config["model"].get("query_prefix", "")
        db = lancedb.connect(config["lancedb"]["path"])
        resources["table"] = db.open_table(config["lancedb"]["table_name"])

    if args.mode in ("bm25", "hybrid"):
        print("BM25インデックス構築中...")
        chunks = load_chunks(args.chunks)
        tokenizer = Dictionary().create()
        tokenized = [tokenize(tokenizer, c["text"]) for c in chunks]
        resources["chunks"] = chunks
        resources["tokenizer"] = tokenizer
        resources["bm25"] = BM25Okapi(tokenized)

    # 評価実行
    queries = load_test_queries(args.query_file)
    print(f"\n評価開始: mode={args.mode}, queries={len(queries)}")
    result = evaluate(args.mode, queries, resources)

    # 結果表示
    print(f"\n===== 結果 =====")
    print(f"モード: {args.mode}")
    if model_name:
        print(f"モデル: {model_name}")
    print(f"クエリ数: {result['num_queries']}")
    print(f"\n{'k':<5}{'Recall@k':<12}{'Precision@k':<14}")
    print("-" * 30)
    for k in K_VALUES:
        print(
            f"{k:<5}"
            f"{result[f'recall_at_{k}']:.3f}       "
            f"{result[f'precision_at_{k}']:.3f}"
        )
    print(f"\nMRR:       {result['mrr']:.3f}")
    print(f"平均検索時間: {result['avg_search_time_ms']:.1f}ms")

    # 正解順位の分布
    print(f"\n----- 正解順位の分布 -----")
    rank_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, "6-10": 0, "圏外": 0}
    for pq in result["per_query"]:
        rr = pq["reciprocal_rank"]
        if rr == 0:
            rank_counts["圏外"] += 1
        else:
            r = round(1 / rr)
            if r <= 5:
                rank_counts[r] += 1
            else:
                rank_counts["6-10"] += 1
    for key, count in rank_counts.items():
        bar = "█" * count
        print(f"{str(key):>5}位: {bar} ({count})")

    # JSON保存
    saved = save_result(result, args.mode, model_name)
    print(f"\n結果保存: {saved}")


if __name__ == "__main__":
    main()
