"""baseline vs ColBERT リランク: 全クエリの順位変動を可視化する。

目的:
  - JaColBERTv2.5 で Recall@1 が 0.650→0.850 に改善した内訳を確認
  - どのクエリで正解チャンクの順位が上がった/下がったかを特定
  - cross-encoder (xsmall) との比較も並記して勝因を分析

使い方:
    cd src && python rerank_diagnose_colbert.py
"""

import lancedb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from sudachipy import Dictionary

from ragatouille import RAGPretrainedModel

from bm25_search import load_chunks, search_bm25, tokenize
from evaluate import load_config, load_test_queries
from hybrid_search import rrf_fusion
from rerank import rerank
from rerank_experiment import colbert_rerank
from search import search as vector_search

EMBED_CONFIG = "config/embedding_e5_small.yaml"
CHUNKS_PATH = "data/processed/chunks_256_ov0.jsonl"
TABLE_NAME = "chunks_256_ov0_e5_small"
RETRIEVE_K = 20
RRF_K = 60
M = 10
TOP_K = 10


def first_correct_rank(results, correct_ids, k=TOP_K):
    """正解記事が最初に出現する順位を返す。見つからなければ None。"""
    for i, r in enumerate(results[:k], 1):
        if r["article_id"] in correct_ids:
            return i
    return None


def main():
    queries = load_test_queries("eval/test_queries.yaml")
    tokenizer = Dictionary().create()

    config = load_config(EMBED_CONFIG)
    embed_model = SentenceTransformer(
        config["model"]["name"],
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )
    query_prefix = config["model"].get("query_prefix", "")

    db = lancedb.connect("data/lance")
    table = db.open_table(TABLE_NAME)

    chunks = load_chunks(CHUNKS_PATH)
    tokenized = [tokenize(tokenizer, c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    print("ColBERT モデルロード中...")
    colbert_model = RAGPretrainedModel.from_pretrained("answerdotai/JaColBERTv2.5")

    print("Cross-encoder (xsmall) モデルロード中...")
    ce_model = CrossEncoder(
        "hotchpotch/japanese-reranker-cross-encoder-xsmall-v1", max_length=512
    )

    print()

    # --- 全クエリ比較 ---
    improved = []   # ColBERT で順位が上がった
    degraded = []   # ColBERT で順位が下がった
    unchanged = []  # 変化なし

    for q in queries:
        if "relevant_articles" not in q:
            continue
        correct_ids = q["relevant_articles"]

        # hybrid baseline
        vec = vector_search(q["query"], embed_model, table, query_prefix, top_k=RETRIEVE_K)
        bm = search_bm25(q["query"], bm25, chunks, tokenizer, top_k=RETRIEVE_K)
        candidates = rrf_fusion(vec, bm, k=RRF_K, top_k=M)
        baseline = rrf_fusion(vec, bm, k=RRF_K, top_k=TOP_K)

        # ColBERT rerank
        colbert_results = colbert_rerank(q["query"], candidates, colbert_model, top_k=TOP_K)

        # Cross-encoder rerank
        ce_results = rerank(q["query"], candidates, ce_model, top_k=TOP_K)

        base_rank = first_correct_rank(baseline, correct_ids)
        colbert_rank = first_correct_rank(colbert_results, correct_ids)
        ce_rank = first_correct_rank(ce_results, correct_ids)

        entry = {
            "query": q["query"],
            "correct_ids": correct_ids,
            "base_rank": base_rank,
            "colbert_rank": colbert_rank,
            "ce_rank": ce_rank,
            "baseline": baseline,
            "colbert_results": colbert_results,
            "ce_results": ce_results,
        }

        if (base_rank or 999) > (colbert_rank or 999):
            improved.append(entry)
        elif (base_rank or 999) < (colbert_rank or 999):
            degraded.append(entry)
        else:
            unchanged.append(entry)

    # --- サマリー ---
    print("=" * 90)
    print("順位変動サマリー (baseline → ColBERT)")
    print("=" * 90)
    print(f"  改善: {len(improved)} クエリ")
    print(f"  悪化: {len(degraded)} クエリ")
    print(f"  変化なし: {len(unchanged)} クエリ")
    print()

    # --- 全クエリ一覧 ---
    all_entries = improved + degraded + unchanged
    print(f"{'クエリ':<40} {'base':>5} {'ColBERT':>8} {'CE(xs)':>7}  変動")
    print("-" * 80)
    for e in all_entries:
        br = e["base_rank"] or "圏外"
        cr = e["colbert_rank"] or "圏外"
        cer = e["ce_rank"] or "圏外"

        if e in improved:
            tag = "↑ 改善"
        elif e in degraded:
            tag = "↓ 悪化"
        else:
            tag = "→ 同"

        q_short = e["query"][:38]
        print(f"  {q_short:<38} {str(br):>5} {str(cr):>8} {str(cer):>7}  {tag}")

    # --- 改善ケース詳細 ---
    if improved:
        print()
        print("=" * 90)
        print("改善ケース詳細 (ColBERT で正解が上位に)")
        print("=" * 90)
        for e in improved:
            _print_detail(e)

    # --- 悪化ケース詳細 ---
    if degraded:
        print()
        print("=" * 90)
        print("悪化ケース詳細 (ColBERT で正解が下降)")
        print("=" * 90)
        for e in degraded:
            _print_detail(e)


def _build_rank_map(results, correct_ids):
    """正解記事のチャンクが結果リストの何位にいるかを返す。
    Returns: {chunk_index: (rank, score, text)} 正解記事のチャンクのみ
    """
    rank_map = {}
    for i, r in enumerate(results, 1):
        if r["article_id"] in correct_ids:
            ci = r.get("chunk_index", "?")
            score = r.get("_rerank_score")
            rank_map[ci] = (i, score, r["text"])
    return rank_map


def _print_detail(e):
    """1クエリの baseline / ColBERT / CE top-K を並べて表示。"""
    print()
    print(f"クエリ: {e['query']}")
    print(f"正解: {e['correct_ids']}")
    print(f"順位: baseline={e['base_rank'] or '圏外'} → ColBERT={e['colbert_rank'] or '圏外'} / CE(xsmall)={e['ce_rank'] or '圏外'}")
    print()

    for label, results in [
        ("baseline", e["baseline"]),
        ("ColBERT", e["colbert_results"]),
        ("CE(xsmall)", e["ce_results"]),
    ]:
        print(f"  --- {label} top-{min(5, len(results))} ---")
        for i, r in enumerate(results[:5], 1):
            mark = "**" if r["article_id"] in e["correct_ids"] else "  "
            score = r.get("_rerank_score")
            score_str = f" score={score:+.3f}" if score is not None else ""
            text_preview = r["text"][:60].replace("\n", " ")
            print(f"    {i}. {mark} [{r['article_id'][:10]}] c{r.get('chunk_index', '?'):>2}{score_str}  {text_preview}")
        print()

    # --- 正解記事チャンクの順位比較 ---
    base_map = _build_rank_map(e["baseline"], e["correct_ids"])
    colbert_map = _build_rank_map(e["colbert_results"], e["correct_ids"])
    ce_map = _build_rank_map(e["ce_results"], e["correct_ids"])

    all_chunks = sorted(set(base_map) | set(colbert_map) | set(ce_map))
    if not all_chunks:
        return

    print(f"  --- 正解記事チャンク 順位マップ (top-{TOP_K}内のみ) ---")
    print(f"    {'chunk':>6}  {'baseline':>9}  {'ColBERT':>9}  {'CE(xs)':>9}  変動         テキスト冒頭")
    print(f"    {'-'*95}")
    for ci in all_chunks:
        b_rank = base_map[ci][0] if ci in base_map else None
        c_rank = colbert_map[ci][0] if ci in colbert_map else None
        ce_rank = ce_map[ci][0] if ci in ce_map else None

        b_str = f"{b_rank:>2}位" if b_rank else "  -"
        c_str = f"{c_rank:>2}位" if c_rank else "  -"
        ce_str = f"{ce_rank:>2}位" if ce_rank else "  -"

        # 変動表示 (baseline → ColBERT)
        if b_rank and c_rank:
            diff = b_rank - c_rank
            if diff > 0:
                delta = f"↑+{diff}"
            elif diff < 0:
                delta = f"↓{diff}"
            else:
                delta = "  ="
        elif c_rank and not b_rank:
            delta = "↑NEW"
        elif b_rank and not c_rank:
            delta = "↓OUT"
        else:
            delta = "  -"

        # テキスト冒頭（どれかから取得）
        text = ""
        for m in [base_map, colbert_map, ce_map]:
            if ci in m:
                text = m[ci][2][:50].replace("\n", " ")
                break

        print(f"    c{ci:>4}  {b_str:>9}  {c_str:>9}  {ce_str:>9}  {delta:<10}  {text}")
    print()


if __name__ == "__main__":
    main()
