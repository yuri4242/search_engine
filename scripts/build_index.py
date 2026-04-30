import json
import yaml
import lancedb
from langchain_core.documents import Document
from langchain_community.vectorstores import LanceDB
from src.embeddings import E5Embeddings

def main():
    config = yaml.safe_load(open("config/langchain_rag.yaml"))

    embeddings = E5Embeddings(
            model_name=config["embedding"]["model"],
        encode_kwargs={"normalize_embeddings": config["embedding"]["normalize"]}
    )

    documents = []
    with open(config["bm25"]["chunks_path"], encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            documents.append(Document(
                page_content=chunk["text"],
                metadata={
                    "article_id": chunk["article_id"],
                    "chunk_index": chunk["chunk_index"],
                    "article_title": chunk["article_title"],
                    "article_url": chunk["article_url"],
                },
            ))

    print(f"Document数: {len(documents)}")
    print("LanceDBへ保存中...")

    db = lancedb.connect(config["vectorstore"]["path"])
    LanceDB.from_documents(
        documents,
        embedding=embeddings,
        connection=db,
        table_name=config["vectorstore"]["table_name"],
    )
    print(f"完了: テーブル　'{config['vectorstore']['table_name']}'")

if __name__ == "__main__":
    main()
