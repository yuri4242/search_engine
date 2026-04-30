"""HyDE（Hypothetical Document Embeddings）: LLMで仮回答を生成し、ベクトル検索を改善する。

通常の検索:  クエリ → query_prefix付きでベクトル化 → 検索
HyDE:       クエリ → LLMが仮回答生成 → passage_prefix付きでベクトル化 → 検索

仮回答はクエリより文書に近い文体になるため、passage_prefix で埋め込む。
これにより検索対象チャンクとベクトル空間上で近くなる。

使い方:
    from hyde import generate_hypothetical, hyde_vector_search
"""

import yaml

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer


def load_hyde_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_hypothetical(
    query: str,
    llm: Llama,
    prompt_template: str,
    max_tokens: int = 200,
    temperature: float = 0.7,
) -> str:
    """
    LLMで仮回答を生成する（HyDE用）。

    Args:
        query: 検索クエリ
        llm: ロード済みの Llama モデル
        prompt_template: {query} プレースホルダを含むプロンプトテンプレート
        max_tokens: 生成する最大トークン数
        temperature: 生成の温度（高いほど多様、低いほど保守的）

    Returns:
        生成された仮回答テキスト
    """
    prompt = prompt_template.format(query=query)
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["\n\n"],
        echo=False,
    )
    return output["choices"][0]["text"].strip()


def hyde_vector_search(
    query: str,
    hypothetical_answer: str,
    model: SentenceTransformer,
    table,
    passage_prefix: str,
    top_k: int = 5,
) -> list[dict]:
    """
    HyDEベクトル検索: 仮回答をpassage_prefix付きでベクトル化し、LanceDBで検索する。

    Args:
        query: 元のクエリ（ログ用、検索自体には使わない）
        hypothetical_answer: LLMが生成した仮回答
        model: Embeddingモデル
        table: LanceDBテーブル
        passage_prefix: パッセージ用prefix（例: "passage: "）
        top_k: 上位何件を返すか

    Returns:
        検索結果のリスト
    """
    # 仮回答をpassage_prefixでベクトル化（query_prefixではない）
    text = passage_prefix + hypothetical_answer
    query_vector = model.encode([text])[0]
    results = table.search(query_vector).distance_type("cosine").limit(top_k).to_list()
    return results
