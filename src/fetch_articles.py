"""Qiita APIから記事を取得してJSONLファイルに保存する。"""

import json
import os
import time
from pathlib import Path

import requests


def fetch_articles(
    query: str = "tag:Python",
    total: int = 100,
    per_page: int = 20,
    output_path: str = "data/raw/articles.jsonl",
) -> int:
    """Qiita API v2から記事を取得してJSONLに保存する。

    Args:
        query: 検索クエリ（例: "tag:Python", "tag:機械学習"）
        total: 取得する記事の総数
        per_page: 1リクエストあたりの取得数（最大100）
        output_path: 保存先のJSONLファイルパス

    Returns:
        実際に保存した記事数
    """
    token = os.environ.get("QIITA_TOKEN")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    saved = 0
    pages = (total + per_page - 1) // per_page  # 切り上げ

    with open(output_path, "w", encoding="utf-8") as f:
        for page in range(1, pages + 1):
            url = "https://qiita.com/api/v2/items"
            params = {
                "page": page,
                "per_page": per_page,
                "query": query,
            }

            response = requests.get(url, headers=headers, params=params)

            # レートリミット情報を表示
            remaining = response.headers.get("Rate-Remaining", "?")
            print(f"  page {page}/{pages} ... {response.status_code} (残り{remaining}回)")

            if response.status_code != 200:
                print(f"  エラー: {response.status_code} {response.text[:200]}")
                break

            articles = response.json()
            if not articles:
                print("  記事がこれ以上ありません")
                break

            for article in articles:
                if saved >= total:
                    break
                f.write(json.dumps(article, ensure_ascii=False) + "\n")
                saved += 1

            if saved >= total:
                break

            # レートリミット対策: リクエスト間に待機
            time.sleep(1)

    print(f"\n完了: {saved}件を {output_path} に保存しました")
    return saved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qiita APIから記事を取得する")
    parser.add_argument("--query", default="tag:Python", help="検索クエリ（デフォルト: tag:Python）")
    parser.add_argument("--total", type=int, default=100, help="取得件数（デフォルト: 100）")
    parser.add_argument("--output", default="data/raw/articles.jsonl", help="出力先ファイル")

    args = parser.parse_args()
    fetch_articles(query=args.query, total=args.total, output_path=args.output)
