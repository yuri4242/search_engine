from langchain_huggingface import HuggingFaceEmbeddings

class E5Embeddings(HuggingFaceEmbeddings):
    """e5系の query: / passage: prefix を自動付与する Embedding。"""

    query_prefix: str = "query: "
    passage_prefix: str = "passage: "

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self.passage_prefix + t for t in texts]
        return super().embed_documents(prefixed)

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(self.query_prefix + text)
