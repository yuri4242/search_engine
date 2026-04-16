"""HyDE仮回答プリコンピュート: LLMで全テストクエリの仮回答を事前生成する。

RAM問題の回避（LLM→生成→アンロード→検索パイプライン）と
実験の再現性（同じ仮回答で複数パターン比較）のために事前生成する。

使い方:
    uv run python src/hyde_generate.py --model models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf --output data/processed/hyde_answers_qwen7b.jsonl
    uv run python src/hyde_generate.py --model models/qwen2.5-3b-instruct-q4_k_m.gguf --output data/processed/hyde_answers_qwen3b.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import yaml
from llama_cpp import Llama

from evaluate import load_test_queries


def load_hyde_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="HyDE仮回答プリコンピュート")
    parser.add_argument("--model", required=True, help="GGUFモデルファイルのパス")
    parser.add_argument("--config", default="config/llm_hyde.yaml", help="HyDE設定ファイル")
    parser.add_argument("--queries", default="eval/test_queries.yaml", help="テストクエリ")
    parser.add_argument("--output", required=True, help="出力JSONLファイル")
    parser.add_argument("--n-threads", type=int, default=8)
    args = parser.parse_args()

    config = load_hyde_config(args.config)
    llm_conf = config["llm"]
    hyde_conf = config["hyde"]
    prompt_template = hyde_conf["prompt_template"]

    queries = load_test_queries(args.queries)
    print(f"テストクエリ数: {len(queries)}")
    print(f"モデル: {args.model}")

    # LLMロード
    print("LLMロード中...")
    load_start = time.time()
    llm = Llama(
        model_path=args.model,
        n_ctx=llm_conf.get("n_ctx", 512),
        n_threads=args.n_threads,
        verbose=False,
    )
    print(f"ロード完了: {time.time() - load_start:.1f}秒")
    print()

    # 全クエリで仮回答を生成
    results = []
    total_start = time.time()
    for i, q in enumerate(queries, 1):
        query = q["query"]
        prompt = prompt_template.format(query=query)

        gen_start = time.time()
        output = llm(
            prompt,
            max_tokens=llm_conf.get("max_tokens", 200),
            temperature=llm_conf.get("temperature", 0.7),
            stop=["\n\n"],
            echo=False,
        )
        gen_time = time.time() - gen_start

        answer = output["choices"][0]["text"].strip()
        tokens = output["usage"]["completion_tokens"]

        print(f"[{i:2}/{len(queries)}] {query[:40]}...")
        print(f"  {tokens}tok / {gen_time:.1f}秒 / {answer[:80]}")

        results.append({
            "query": query,
            "hypothetical_answer": answer,
            "tokens": tokens,
            "generation_time_s": round(gen_time, 2),
        })

    total_time = time.time() - total_start

    # LLMアンロード
    del llm

    # 保存
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n合計生成時間: {total_time:.1f}秒")
    print(f"平均生成時間: {total_time / len(results):.1f}秒/クエリ")
    print(f"保存先: {args.output}")


if __name__ == "__main__":
    main()
