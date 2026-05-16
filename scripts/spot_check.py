"""LCEL chain を 7 クエリで spot check し、結果を JSONL に保存する。

chain は1回だけ初期化し、各クエリで invoke する（初期化コストの償却）。
出力: eval/spot_check_lcel.jsonl
"""

import json
import time
from pathlib import Path

import yaml

from src.rag import build_rag_chain


N_QUERIES = 7
OUTPUT_PATH = Path("eval/spot_check_lcel.jsonl")


def main():
    config = yaml.safe_load(open("config/langchain_rag.yaml"))

    with open("eval/test_queries.yaml", encoding="utf-8") as f:
        all_queries = yaml.safe_load(f)
    queries = all_queries[:N_QUERIES]

    print(f"初期化中...")
    init_start = time.time()
    chain = build_rag_chain(config)
    print(f"初期化完了 ({time.time() - init_start:.1f}秒)\n")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, q in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] {q['query']}")
            start = time.time()
            result = chain.invoke(q["query"])
            elapsed = time.time() - start

            record = {
                "query": q["query"],
                "relevant_articles": q.get("relevant_articles", []),
                "answer": result["answer"].strip(),
                "retrieved_article_ids": [d.metadata["article_id"] for d in result["context"]],
                "retrieved_titles": [d.metadata["article_title"] for d in result["context"]],
                "elapsed_sec": round(elapsed, 1),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            records.append(record)

            print(f"  回答 ({elapsed:.1f}秒): {record['answer'][:80]}...")
            print(f"  根拠 ids: {record['retrieved_article_ids']}")
            print()

    total_time = sum(r["elapsed_sec"] for r in records)
    correct_in_top3 = sum(
        1 for r in records
        if any(aid in r["retrieved_article_ids"] for aid in r["relevant_articles"])
    )
    print("=== サマリ ===")
    print(f"クエリ数: {len(records)}")
    print(f"総生成時間: {total_time:.0f}秒")
    print(f"正解記事 top-3 ヒット: {correct_in_top3}/{len(records)}")
    print(f"出力先: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
