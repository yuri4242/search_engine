"""チャンクをベクトル化してLanceDBに保存する。

使い方:
    python src/embed_chunks.py --config config/embedding_ruri_small.yaml
"""

import argparse
import json
import time
from pathlib import Path

import lancedb
import yaml
from sentence_transformers import SentenceTransformer


def load_config(config_path: str) -> dict:
    """YAML設定ファイルを読む。"""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_chunks(chunks_path: str) -> list[dict]:
    """JSONLからチャンクを読んでリストで返す。"""
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def embed_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
    passage_prefix: str,
) -> list[dict]:
    """
    チャンクをベクトル化する。

    Args:
        chunks: チャンクのリスト（各要素に "text" が入っている）
        model: ロード済みのSentenceTransformerモデル
        passage_prefix: 各テキストの先頭に付ける prefix（例: "passage: "）

    Returns:
        各チャンクに "vector" フィールドを追加したリスト
    """
    # 1. 各チャンクのテキストに prefix を付けて、文字列のリストを作る
    texts = [passage_prefix + chunk["text"] for chunk in chunks]

    # 2. model.encode(texts, show_progress_bar=True) でまとめてベクトル化する
    #    normalize_embeddings=True で単位ベクトル化（内積=コサイン類似度になる）
    vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # 3. 各チャンクに "vector" フィールドを追加して返す
    for chunk, vec in zip(chunks, vectors):
        chunk["vector"] = vec.tolist()
    return chunks

def save_to_lancedb(chunks_with_vectors: list[dict], db_path: str, table_name: str) -> int:
    """LanceDBにチャンクを保存する。"""
    db = lancedb.connect(db_path)

    # 既存のテーブルがあれば削除してから作り直す（再実行可能にする）
    if table_name in db.list_tables().tables:
        db.drop_table(table_name)

    table = db.create_table(table_name, data=chunks_with_vectors)
    return len(table)


def main():
    parser = argparse.ArgumentParser(description="チャンクをベクトル化してLanceDBに保存")
    parser.add_argument("--config", required=True, help="設定ファイル（YAML）のパス")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl",
                        help="入力のチャンクJSONL")
    args = parser.parse_args()

    # 設定読み込み
    config = load_config(args.config)
    model_name = config["model"]["name"]
    passage_prefix = config["model"].get("passage_prefix", "")
    trust_remote_code = config["model"].get("trust_remote_code", False)
    db_path = config["lancedb"]["path"]
    table_name = config["lancedb"]["table_name"]

    print(f"設定: {args.config}")
    print(f"  モデル: {model_name}")
    print(f"  passage_prefix: {repr(passage_prefix)}")
    print(f"  保存先: {db_path}/{table_name}")
    print()

    # チャンク読み込み
    print("チャンク読み込み中...")
    chunks = load_chunks(args.chunks)
    print(f"  チャンク数: {len(chunks)}")
    print()

    # モデルロード
    print(f"モデルをロード中: {model_name}")
    start = time.time()
    model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
    print(f"  ロード完了（{time.time() - start:.1f}秒）")
    print()

    # ベクトル化
    print("ベクトル化中...")
    start = time.time()
    chunks_with_vectors = embed_chunks(chunks, model, passage_prefix)
    print(f"  ベクトル化完了（{time.time() - start:.1f}秒）")
    print(f"  1チャンクあたり: {(time.time() - start) / len(chunks) * 1000:.1f}ms")
    print()

    # LanceDBに保存
    print("LanceDBに保存中...")
    start = time.time()
    count = save_to_lancedb(chunks_with_vectors, db_path, table_name)
    print(f"  保存完了（{time.time() - start:.1f}秒）")
    print(f"  登録件数: {count}")


if __name__ == "__main__":
    main()
