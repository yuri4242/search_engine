"""全記事をチャンク分割して data/processed/chunks.jsonl に保存する。"""

import json
from pathlib import Path

from chunker import chunk_article


def build_chunks(
    input_path: str = "data/raw/articles.jsonl",
    output_path: str = "data/processed/chunks.jsonl",
    chunk_size: int = 400,
) -> int:
    """全記事をチャンク分割してJSONLに保存する。

    Args:
        input_path: 入力の記事JSONLファイル
        output_path: 出力のチャンクJSONLファイル
        chunk_size: チャンクの文字数

    Returns:
        出力したチャンクの総数
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    article_count = 0
    chunk_count = 0

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            article = json.loads(line)
            chunks = chunk_article(article, chunk_size=chunk_size)
            for chunk in chunks:
                fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                chunk_count += 1
            article_count += 1

    print(f"記事数: {article_count}")
    print(f"チャンク数: {chunk_count}")
    print(f"平均チャンク数/記事: {chunk_count / article_count:.1f}")
    print(f"出力先: {output_path}")
    return chunk_count


if __name__ == "__main__":
    build_chunks()
