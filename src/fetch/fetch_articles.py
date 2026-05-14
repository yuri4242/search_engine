"""Qiita APIから記事を取得してJSONLファイルに保存する。

2つの取得モードを組み合わせる:
  1. --include-ids-from で指定した YAML から正解 ID を抽出 → /items/:id でピンポイント取得
  2. --query のタグ検索で新着順に残り件数を埋める（ピンポイント分と重複したらスキップ）
"""

import json
import os
import time
from pathlib import Path

import requests
import yaml


def extract_ids_from_queries(yaml_path: str) -> list[str]:
    """test_queries.yaml の relevant_articles から重複排除したID列を返す。"""
    with open(yaml_path, encoding="utf-8") as f:
        queries = yaml.safe_load(f)
    ids: list[str] = []
    seen: set[str] = set()
    for q in queries:
        for aid in q.get("relevant_articles", []):
            if aid not in seen:
                seen.add(aid)
                ids.append(aid)
    return ids


def fetch_by_ids(ids: list[str], headers: dict) -> list[dict]:
    """指定ID の記事を個別フェッチ。失敗した ID は警告して continue。"""
    articles = []
    for i, article_id in enumerate(ids, 1):
        url = f"https://qiita.com/api/v2/items/{article_id}"
        response = requests.get(url, headers=headers)
        remaining = response.headers.get("Rate-Remaining", "?")
        print(f"  id {i}/{len(ids)}: {article_id} ... {response.status_code} (残り{remaining}回)")
        if response.status_code == 200:
            articles.append(response.json())
        else:
            print(f"    ⚠ 取得失敗: {response.text[:200]}")
        time.sleep(1)
    return articles


def fetch_articles(
    query: str = "tag:RAG",
    total: int = 100,
    per_page: int = 20,
    output_path: str = "data/raw/articles.jsonl",
    include_ids_from: str | None = None,
) -> int:
    """Qiita API v2 から記事を取得して JSONL に保存する。

    include_ids_from が指定されたら、その YAML の relevant_articles を
    ピンポイント取得し、残りをタグ検索で埋める。
    """
    token = os.environ.get("QIITA_TOKEN")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        print("⚠ QIITA_TOKEN が未設定。レートリミットが厳しくなります（60/h）。")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 1. ピンポイント取得
    pinpoint_articles: list[dict] = []
    pinpoint_ids: set[str] = set()
    if include_ids_from:
        ids = extract_ids_from_queries(include_ids_from)
        print(f"正解 ID をフェッチ（{len(ids)}件、source={include_ids_from}）")
        pinpoint_articles = fetch_by_ids(ids, headers)
        pinpoint_ids = {a["id"] for a in pinpoint_articles}
        print(f"  取得済み: {len(pinpoint_articles)}件\n")

    # 2. タグ検索で残りを埋める
    remaining_total = total - len(pinpoint_articles)
    tag_articles: list[dict] = []
    if remaining_total > 0:
        print(f"タグ検索で残り {remaining_total} 件を取得（query={query}）")
        page = 1
        max_pages = 20  # 安全弁: 重複排除で足りなくなり続けたら止める
        while len(tag_articles) < remaining_total and page <= max_pages:
            url = "https://qiita.com/api/v2/items"
            params = {"page": page, "per_page": per_page, "query": query}
            response = requests.get(url, headers=headers, params=params)
            remaining = response.headers.get("Rate-Remaining", "?")
            print(f"  page {page}: {response.status_code} (残り{remaining}回)")
            if response.status_code != 200:
                print(f"    エラー: {response.text[:200]}")
                break
            articles = response.json()
            if not articles:
                print("    これ以上記事がありません")
                break
            for article in articles:
                if article["id"] in pinpoint_ids:
                    continue
                tag_articles.append(article)
                if len(tag_articles) >= remaining_total:
                    break
            page += 1
            time.sleep(1)

    # 3. 書き出し
    saved = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for article in pinpoint_articles + tag_articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")
            saved += 1

    print(f"\n完了: {saved}件（pinpoint={len(pinpoint_articles)}, tag={len(tag_articles)}）")
    print(f"出力先: {output_path}")
    return saved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qiita APIから記事を取得する")
    parser.add_argument("--query", default="tag:RAG", help="タグ検索クエリ")
    parser.add_argument("--total", type=int, default=100, help="取得件数")
    parser.add_argument("--output", default="data/raw/articles.jsonl", help="出力先 JSONL")
    parser.add_argument("--include-ids-from", default=None, help="正解 ID を抽出する YAML ファイル")
    args = parser.parse_args()

    fetch_articles(
        query=args.query,
        total=args.total,
        output_path=args.output,
        include_ids_from=args.include_ids_from,
    )
