import yaml

from src.rag import build_ensemble_retriever

def main():
    config = yaml.safe_load(open("config/langchain_rag.yaml"))
    retriever = build_ensemble_retriever(config)

    with open("eval/test_queries.yaml", encoding="utf-8") as f:
        queries = yaml.safe_load(f)

    K_VALUES = [1, 3, 5, 10]
    recall_at_k = {k: 0 for k in K_VALUES}
    reciprocal_ranks = []

    for q in queries:
        query = q["query"]
        relevant = set(q["relevant_articles"])

        results = retriever.invoke(query)
        article_ids = [d.metadata["article_id"] for d in results]

        for k in K_VALUES:
            if any(aid in relevant for aid in article_ids[:k]):
                recall_at_k[k] += 1

        rr = 0
        for rank, aid in enumerate(article_ids, 1):
            if aid in relevant:
                rr =  1 / rank
                break
        reciprocal_ranks.append(rr)

    n = len(queries)
    print(f"クエリ数: {n}")
    for k in K_VALUES:
        print(f"Recall@{k}: {recall_at_k[k]/n:.3f}")
    print(f"MRR: {sum(reciprocal_ranks)/n:.3f}")

if __name__ == "__main__":
    main()
