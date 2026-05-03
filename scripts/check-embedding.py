import lancedb
import numpy as np

from src.embeddings import E5Embeddings

db = lancedb.connect("data/lance")
table = db.open_table("chunks_256_ov0_e5_small_lc")
row = table.to_pandas().head(1).to_dict("records")[0]
text = row["text"]
existing_vec = np.array(row["vector"])

emb = E5Embeddings(
    model_name = "intfloat/multilingual-e5-small",
    encode_kwargs={"normalize_embeddings": True}
)

new_vec_with = np.array(emb.embed_documents([text])[0])

new_vec_without = np.array(emb._client.encode([text], normalize_embeddings=True)[0])

print(f"既存ベクトル先頭: {existing_vec[:5]}")
print(f"passage付で再計算: {new_vec_with[:5]}")
print(f"prefix無しで計算: {new_vec_without[:5]}")
print()
print(f"既存 vs passage付: {'一致' if np.allclose(existing_vec, new_vec_with, atol=1e-5) else '不一致'}")
print(f"既存 vs prefix無: {'一致' if np.allclose(existing_vec, new_vec_without, atol=1e-5) else '不一致'}")
