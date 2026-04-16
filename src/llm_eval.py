"""LLMモデル品質スクリーニング: HyDE用の仮回答生成品質を比較する。

5つの代表クエリで仮回答を生成し、以下を計測:
- 生成速度（トークン/秒）
- RAM使用量
- 出力品質（目視確認用に保存）

使い方:
    uv run python src/llm_eval.py
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import psutil
import yaml

from llama_cpp import Llama

# 代表クエリ（多様なタイプを選択）
EVAL_QUERIES = [
    {"query": "BM25 ベクトル検索 RRF ハイブリッドRAG", "type": "キーワード的"},
    {"query": "専門用語がRAGでヒットしない問題の解決策", "type": "言い換え"},
    {"query": "RAGとは何か 初心者向け わかりやすく", "type": "初心者向け"},
    {"query": "RAGにナレッジグラフを組み合わせると何ができるようになる？", "type": "質問文"},
    {"query": "embeddingモデルとチャンクサイズで検索精度がどう変わるか実測", "type": "実験系"},
]

PROMPT_TEMPLATE = """以下の検索クエリに対して、Qiitaの技術記事に書いてありそうな回答を100〜200文字で書いてください。
正確さよりも、関連する技術用語や表現を含むことが重要です。

クエリ: {query}
回答:"""

# 評価するモデル
MODELS = [
    {
        "name": "Qwen2.5-7B-Instruct Q4_K_M",
        "path": "models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        "n_ctx": 512,
        "n_threads": 8,
    },
    {
        "name": "gemma-2-2b-it Q4_K_M",
        "path": "models/gemma-2-2b-it-Q4_K_M.gguf",
        "n_ctx": 512,
        "n_threads": 8,
    },
]


def get_ram_usage_mb():
    """現在のプロセスのRAM使用量をMBで返す。"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def eval_model(model_config):
    """1モデルを評価する。"""
    print(f"\n{'='*70}")
    print(f"モデル: {model_config['name']}")
    print(f"パス: {model_config['path']}")
    print(f"{'='*70}")

    if not Path(model_config["path"]).exists():
        print(f"  ⚠ モデルファイルが見つかりません。スキップ。")
        return None

    ram_before = get_ram_usage_mb()
    print(f"  RAM (ロード前): {ram_before:.0f} MB")

    print(f"  モデルロード中...")
    load_start = time.time()
    llm = Llama(
        model_path=model_config["path"],
        n_ctx=model_config["n_ctx"],
        n_threads=model_config["n_threads"],
        verbose=False,
    )
    load_time = time.time() - load_start

    ram_after_load = get_ram_usage_mb()
    print(f"  ロード完了: {load_time:.1f}秒")
    print(f"  RAM (ロード後): {ram_after_load:.0f} MB (+{ram_after_load - ram_before:.0f} MB)")
    print()

    results = []
    for q in EVAL_QUERIES:
        prompt = PROMPT_TEMPLATE.format(query=q["query"])

        print(f"  クエリ [{q['type']}]: {q['query']}")

        gen_start = time.time()
        output = llm(
            prompt,
            max_tokens=200,
            temperature=0.7,
            stop=["\n\n"],
            echo=False,
        )
        gen_time = time.time() - gen_start

        text = output["choices"][0]["text"].strip()
        tokens_generated = output["usage"]["completion_tokens"]
        tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0

        print(f"  回答 ({tokens_generated}tok, {gen_time:.1f}秒, {tokens_per_sec:.1f}tok/s):")
        print(f"    {text[:200]}")
        print()

        results.append({
            "query": q["query"],
            "query_type": q["type"],
            "answer": text,
            "tokens_generated": tokens_generated,
            "generation_time_s": round(gen_time, 2),
            "tokens_per_sec": round(tokens_per_sec, 1),
        })

    # モデルをアンロード
    del llm

    ram_after_unload = get_ram_usage_mb()

    summary = {
        "model_name": model_config["name"],
        "model_path": model_config["path"],
        "load_time_s": round(load_time, 1),
        "ram_model_mb": round(ram_after_load - ram_before),
        "ram_after_unload_mb": round(ram_after_unload),
        "avg_generation_time_s": round(
            sum(r["generation_time_s"] for r in results) / len(results), 1
        ),
        "avg_tokens_per_sec": round(
            sum(r["tokens_per_sec"] for r in results) / len(results), 1
        ),
        "queries": results,
    }

    print(f"  --- サマリー ---")
    print(f"  ロード時間: {summary['load_time_s']}秒")
    print(f"  モデルRAM: {summary['ram_model_mb']} MB")
    print(f"  平均生成時間: {summary['avg_generation_time_s']}秒")
    print(f"  平均速度: {summary['avg_tokens_per_sec']} tok/s")

    return summary


def main():
    print("LLM品質スクリーニング for HyDE")
    print(f"評価クエリ数: {len(EVAL_QUERIES)}")
    print(f"評価モデル数: {len(MODELS)}")

    all_results = []
    for m in MODELS:
        result = eval_model(m)
        if result:
            all_results.append(result)

    # 比較サマリー
    if len(all_results) >= 2:
        print(f"\n{'='*70}")
        print("モデル比較サマリー")
        print(f"{'='*70}")
        print(f"{'モデル':<30} {'ロード':>8} {'RAM':>8} {'生成':>8} {'速度':>10}")
        print("-" * 70)
        for r in all_results:
            print(
                f"{r['model_name']:<30} "
                f"{r['load_time_s']:>6.1f}秒 "
                f"{r['ram_model_mb']:>6}MB "
                f"{r['avg_generation_time_s']:>6.1f}秒 "
                f"{r['avg_tokens_per_sec']:>7.1f}tok/s"
            )

    # 結果保存
    Path("benchmarks").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmarks/llm_eval_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {filename}")


if __name__ == "__main__":
    main()
