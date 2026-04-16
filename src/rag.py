"""RAG回答生成: 検索結果のチャンクをLLMに渡して自然言語で回答する。

パイプライン:
  クエリ → hybrid検索(top-20) → RRF → ColBERTリランク(top-10) → 上位3件をLLMに渡す → 回答生成

使い方:
    uv run python src/rag.py --query "専門用語がRAGでヒットしない問題の解決策"
"""

import argparse
import time

import lancedb
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sudachipy import Dictionary

from llama_cpp import Llama
from ragatouille import RAGPretrainedModel

from bm25_search import load_chunks, search_bm25, tokenize
from hybrid_search import rrf_fusion
from rerank_experiment import colbert_rerank
from search import search as vector_search

# パイプライン設定
EMBED_CONFIG = "config/embedding_e5_small.yaml"
LLM_CONFIG = "config/llm_hyde.yaml"
CHUNKS_PATH = "data/processed/chunks_256_ov0.jsonl"
TABLE_NAME = "chunks_256_ov0_e5_small"
RETRIEVE_K = 20
RRF_K = 60
COLBERT_M = 10
RAG_TOP_K = 3  # LLMに渡すチャンク数

SYSTEM_PROMPT = (
    "以下の情報を元に質問に回答してください。"
    "情報に含まれない内容には「情報が見つかりませんでした」と答えてください。"
)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt(query, chunks):
    """検索結果チャンクとクエリからプロンプトを組み立てる。"""
    context = ""
    for i, c in enumerate(chunks, 1):
        context += f"[{i}] {c['text']}\n\n"

    return f"{SYSTEM_PROMPT}\n\n{context}質問: {query}\n回答:"


def generate_answer(prompt, llm, max_tokens=150):
    """LLMで回答を生成する。"""
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.3,
        stop=["\n\n", "質問:", "質問："],
        echo=False,
    )
    return output["choices"][0]["text"].strip()


def search_pipeline(query, embed_model, table, query_prefix, bm25, chunks, tokenizer, colbert_model):
    """検索パイプライン: hybrid → RRF → ColBERT rerank → top-3"""
    vec = vector_search(query, embed_model, table, query_prefix, top_k=RETRIEVE_K)
    bm = search_bm25(query, bm25, chunks, tokenizer, top_k=RETRIEVE_K)
    candidates = rrf_fusion(vec, bm, k=RRF_K, top_k=COLBERT_M)
    reranked = colbert_rerank(query, candidates, colbert_model, top_k=RAG_TOP_K)
    return reranked


def main():
    parser = argparse.ArgumentParser(description="RAG質問応答")
    parser.add_argument("--query", required=True, help="質問")
    args = parser.parse_args()

    print("初期化中...")
    init_start = time.time()

    # Embedding
    embed_config = load_config(EMBED_CONFIG)
    model_name = embed_config["model"]["name"]
    query_prefix = embed_config["model"].get("query_prefix", "")
    embed_model = SentenceTransformer(
        model_name,
        trust_remote_code=embed_config["model"].get("trust_remote_code", False),
    )

    # LanceDB
    db = lancedb.connect("data/lance")
    table = db.open_table(TABLE_NAME)

    # BM25
    tokenizer = Dictionary().create()
    chunks = load_chunks(CHUNKS_PATH)
    tokenized = [tokenize(tokenizer, c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    # ColBERT
    colbert_model = RAGPretrainedModel.from_pretrained("answerdotai/JaColBERTv2.5")

    # LLM
    llm_config = load_config(LLM_CONFIG)
    llm = Llama(
        model_path=llm_config["llm"]["model_path"],
        n_ctx=1024,
        n_threads=llm_config["llm"].get("n_threads", 8),
        verbose=False,
    )

    init_time = time.time() - init_start
    print(f"初期化完了（{init_time:.1f}秒）\n")

    # 検索
    print(f"[質問] {args.query}")
    print("-" * 70)

    search_start = time.time()
    results = search_pipeline(
        args.query, embed_model, table, query_prefix,
        bm25, chunks, tokenizer, colbert_model,
    )
    search_time = time.time() - search_start

    print(f"\n検索結果（{search_time:.1f}秒、上位{len(results)}件）:")
    for i, r in enumerate(results, 1):
        score = r.get("_rerank_score", 0)
        print(f"  [{i}] {r['article_title'][:60]}")
        print(f"      chunk {r['chunk_index']} / score={score:.3f}")
        print(f"      {r['text'][:80]}...")
        print()

    # 回答生成
    prompt = build_prompt(args.query, results)
    print(f"回答生成中...")
    gen_start = time.time()
    answer = generate_answer(prompt, llm)
    gen_time = time.time() - gen_start

    print(f"\n[回答]（{gen_time:.1f}秒）")
    print(answer)
    print()

    # 根拠
    print("根拠:")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['article_title'][:70]}")
        print(f"      {r['article_url']}")


if __name__ == "__main__":
    main()
