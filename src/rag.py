import yaml
import lancedb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from typing import List

config = yaml.safe_load(open("config/langchain_rag.yaml"))

class E5Embeddings(HuggingFaceEmbeddings):
    query_prefix: str="query: "
    passage_prefix: str="passage: "
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents([self.passage_prefix + t for t in texts])
    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(self.query_prefix + text)

embedding = E5Embeddings(
    model_name=config["embedding"]["model"],
    encode_kwargs={"normalize_embeddings": config["embedding"]["normalize"]},
)

db = lancedb.connect(config["vectorstore"]["path"])
vectorstore = LanceDB(connection=db, table_name=config["vectorstore"]["table_name"], embedding=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": config["vectorstore"]["retriever_k"]})

results = retriever.invoke("RAGのチャンク分割")
for d in results:
    print(d.metadata)
    #print(d.metadata.get("article_id", "?"), d.page_content[:50])

