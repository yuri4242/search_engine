"""Embedding モデルの動作確認（Hello World）。"""

import time

from sentence_transformers import SentenceTransformer


def main():
    print("モデルをロード中...")
    start = time.time()
    model = SentenceTransformer("cl-nagoya/ruri-small", trust_remote_code=True)
    elapsed = time.time() - start
    print(f"  ロード完了（{elapsed:.1f}秒）")
    print()

    # ruri系は instruction prefix が必要
    # query: で始まる文は検索クエリとして、passage: で始まる文はインデックス対象として扱う
    sentences = [
        "query: RAGのチャンク分割について",
        "passage: RAGでは文書を一定の大きさで区切ってベクトル化する",
        "passage: 今日は良い天気です",
    ]

    print("ベクトル化中...")
    start = time.time()
    vectors = model.encode(sentences)
    elapsed = time.time() - start
    print(f"  ベクトル化完了（{elapsed:.2f}秒）")
    print()

    print(f"ベクトルの形状: {vectors.shape}")
    print(f"  → {vectors.shape[0]} 文を、各 {vectors.shape[1]} 次元のベクトルに変換")
    print()

    # 類似度計算（コサイン類似度）
    from sentence_transformers.util import cos_sim

    query_vec = vectors[0:1]
    passage_vecs = vectors[1:]

    sims = cos_sim(query_vec, passage_vecs)
    print("コサイン類似度（クエリ vs 各パッセージ）:")
    for i, sim in enumerate(sims[0]):
        print(f"  「{sentences[i + 1][:40]}...」 → {sim:.3f}")


if __name__ == "__main__":
    main()
