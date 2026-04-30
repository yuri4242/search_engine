def chunk_article(article: dict, chunk_size: int = 400, overlap: int = 0) -> list[dict]:
    """
    記事を文字数で等分割してチャンクのリストを返す。

    Args:
        article: JSONLの1行分（dict）。body, id, title, url などを含む。
        chunk_size: チャンクの文字数（デフォルト：400）

    Returns:
        チャンクのリスト。各要素は以下の形式：
        {
            "article_id": str,
            "article_title": str,
            "article_url": str,
            "chunk_index": int,
            "text": str
        }
    """
    #1. 記事本分と、チャンクにつけるメタ情報を取り出す
    body = article["body"]
    article_id = article["id"]
    article_title = article["title"]
    article_url = article["url"]

    #2. チャンクを入れる空リストを用意
    chunks: list[dict] = []

    #3. body を chunk_size ずつ区切って chunks に追加していく
    stride = chunk_size - overlap
    for chunk_index, i in enumerate(range(0, len(body), stride)):
        text = body[i : i + chunk_size]
        chunks.append({
            "article_id": article_id,
            "article_title": article_title,
            "article_url": article_url,
            "chunk_index": chunk_index,
            "text": text
        })
    return chunks

if __name__ == "__main__":
    import json

    with open("data/raw/articles.jsonl", encoding="utf-8") as f:
        first_article = json.loads(f.readline())

    chunks = chunk_article(first_article)
    print(f"記事タイトル: {first_article['title']}")
    print(f"本分の長さ: {len(first_article['body'])} 文字")
    print(f"チャンク数: {len(chunks)}")
    print()
    for chunk in chunks[:3]:
        print(f"--- chunk_index={chunk['chunk_index']} ---")
        print(chunk["text"][:100] + "...")
        print()
