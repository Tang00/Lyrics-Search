import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

df = pd.read_csv("data/lyrics_cleaned.csv")
embeddings = np.load("sbert_embeddings.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")

queries = [
    "Heartbreak", "Unrequited love", "Jealousy", "Falling in love", "Letting go",
    "Loneliness", "Depression", "Anxiety", "Self-doubt",
    "Betrayal", "Lies", "Revenge",
    "Meaning", "Aging", "Lost",
    "Resilience", "New beginnings", "Self-worth",
    "Misunderstood", "Escape"
]

def sbert_search(query, k=10):
    query_emb = model.encode(query, normalize_embeddings=True)
    scores = util.cos_sim(query_emb, embeddings)[0]
    top_idx = scores.topk(k).indices.tolist()
    return top_idx, scores

rows = []

for q in queries:
    idxs, scores = sbert_search(q, k=10)
    for rank, i in enumerate(idxs, start=1):
        rows.append({
            "query": q,
            "rank": rank,
            "song": df.loc[i, "song"],
            "artist": df.loc[i, "artist"],
            "text": df.loc[i, "text"],
            "score": float(scores[i]),
            "relevance": ""
        })

out_df = pd.DataFrame(rows)
out_df.to_csv("sbert_evaluation_sheet.csv", index=False)
