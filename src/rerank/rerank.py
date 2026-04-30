"""Cross-encoder によるリランキング。

ハイブリッド検索で取得した top-M 候補を、query と chunk のペアで
スコア付けし直し、top-K に絞り込む。

使い方（コア関数）:
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("hotchpotch/japanese-reranker-cross-encoder-xsmall-v1", max_length=512)
    reranked = rerank(query, candidates, model, top_k=5)
"""

from sentence_transformers import CrossEncoder


def rerank(
    query: str,
    candidates: list[dict],
    model: CrossEncoder,
    top_k: int = 5,
) -> list[dict]:
    """
    cross-encoder で candidates を再スコアし、上位 top_k を返す。

    Args:
        query: 検索クエリ
        candidates: ハイブリッド検索結果（各要素に "text" と "article_id" を含む）
        model: sentence_transformers.CrossEncoder
        top_k: 返す件数

    Returns:
        reranked 結果（各要素に "_rerank_score" を付与、スコア降順）
    """
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    scored = []
    for c, s in zip(candidates, scores):
        c_copy = dict(c)
        c_copy["_rerank_score"] = float(s)
        scored.append(c_copy)
    scored.sort(key=lambda x: x["_rerank_score"], reverse=True)
    return scored[:top_k]
