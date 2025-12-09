import pandas as pd
import numpy as np

bm = "manual_evaluations/bm25_evaluation_sheet.csv"
sbert = "manual_evaluations/sbert_evaluation_sheet.csv"

df = pd.read_csv(sbert)

def precision_at_k(relevances, k=10):
    r = np.array(relevances)[:k]
    return np.sum(r > 0) / k

def dcg_at_k(relevances, k=10):
    r = np.array(relevances)[:k]
    discounts = np.log2(np.arange(2, len(r) + 2))
    return np.sum((2**r - 1) / discounts)

def ndcg_at_k(relevances, k=10):
    dcg = dcg_at_k(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0

precisions = []
ndcgs = []

for query in df["query"].unique():
    qdf = df[df["query"] == query].sort_values("rank")
    rels = qdf["relevance"].astype(int).tolist()
    
    precisions.append(precision_at_k(rels, k=10))
    ndcgs.append(ndcg_at_k(rels, k=10))

print(f"Mean Precision@10: {np.mean(precisions):.4f}")
print(f"Mean NDCG@10:      {np.mean(ndcgs):.4f}")
