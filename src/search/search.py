"""クエリをベクトル化してLanceDBで類似検索する。

使い方:
    python src/search.py --config config/embedding_ruri_small.yaml --query "RAGのチャンク分割 最適なサイズ"
    python src/search.py --config config/embedding_ruri_small.yaml --query-file eval/test_queries.yaml
"""

import argparse
import time
from typing import Any

import lancedb
import yaml
from sentence_transformers import SentenceTransformer


def load_config(config_path: str) -> dict:
    """YAML設定ファイルを読む。"""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_test_queries(path: str) -> list[dict]:
    """テストクエリYAMLを読んでリストで返す。"""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def search(
    query: str,
    model: SentenceTransformer,
    table: lancedb.table.Table,
    query_prefix: str,
    top_k: int = 5,
) -> list[dict]:
    """
    クエリを検索して上位top_k件を返す。

    Args:
        query: 検索クエリ（日本語の文字列）
        model: ロード済みのSentenceTransformerモデル
        table: LanceDBのテーブル（open_tableの戻り値）
        query_prefix: クエリに付ける prefix（例: "query: "）
        top_k: 上位何件を返すか

    Returns:
        上位top_k件の検索結果のリスト。各要素には以下が入る:
        {
            "article_id": str,
            "article_title": str,
            "article_url": str,
            "chunk_index": int,
            "text": str,
            "_distance": float,   # LanceDBが自動で追加する距離スコア（小さいほど近い）
        }
    """
    # 1. クエリに prefix を付けてベクトル化する
    #    - model.encode() は通常リストを受け取るが、1件だけなら [query_text] を渡す
    #    - 戻り値は (1, 次元数) の配列なので [0] で取り出して1次元にする
    query_text = query_prefix + query
    query_vector = model.encode([query_text])[0]

    # 2. table.search(vector).limit(top_k).to_list() で類似検索する
    #    - LanceDBは同じ次元のベクトルを渡せばコサイン類似度で検索してくれる
    #    - .to_list() で Python の list[dict] に変換
    results = table.search(query_vector).distance_type("cosine").limit(top_k).to_list()
    return results


def print_results(query: str, results: list[dict]) -> None:
    """検索結果を見やすく表示する。"""
    print(f"\n[クエリ] {query}")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        distance = r.get("_distance", 0)
        print(f"  {i}. ({r['article_id'][:8]}... chunk {r['chunk_index']:2d}) "
              f"距離={distance:.3f}")
        print(f"     {r['article_title'][:70]}")
        print(f"     本文: {r['text'][:80]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="LanceDBで類似検索を行う")
    parser.add_argument("--config", required=True, help="設定ファイル（YAML）")
    parser.add_argument("--query", help="単発クエリ文字列")
    parser.add_argument("--query-file", help="テストクエリYAMLファイル（全クエリを処理）")
    parser.add_argument("--top-k", type=int, default=5, help="上位何件を返すか")
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
    print(f"  query_prefix: {repr(query_prefix)}")
    print(f"  検索対象: {db_path}/{table_name}")


    # モデルロード
    print(f"\nモデルをロード中...")
    start = time.time()
    model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
    print(f"  ロード完了（{time.time() - start:.1f}秒）")

    # LanceDBのテーブルを開く
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    print(f"  テーブル件数: {len(table)}")

    # 検索実行
    if args.query:
        # 単発クエリ
        results = search(args.query, model, table, query_prefix, args.top_k)
        print_results(args.query, results)
    else:
        # YAMLの全クエリを処理
        queries = load_test_queries(args.query_file)
        print(f"\nテストクエリ数: {len(queries)}")
        for q in queries:
            results = search(q["query"], model, table, query_prefix, args.top_k)
            print_results(q["query"], results)
            # 正解記事IDも表示（あれば）
            if "relevant_articles" in q:
                correct_ids = [a[:8] for a in q["relevant_articles"]]
                returned_ids = [r["article_id"][:8] for r in results]
                hit = any(cid in returned_ids for cid in correct_ids)
                print(f"     正解記事: {correct_ids} / ヒット: {'✓' if hit else '✗'}")


if __name__ == "__main__":
    main()
