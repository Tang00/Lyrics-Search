from pyserini.search.lucene import LuceneSearcher
import pandas as pd
import json

queries = [
    "Heartbreak", "Unrequited love", "Jealousy", "Falling in love", "Letting go",
    "Loneliness", "Depression", "Anxiety", "Self-doubt",
    "Betrayal", "Lies", "Revenge",
    "Meaning", "Aging", "Lost",
    "Resilience", "New beginnings", "Self-worth",
    "Misunderstood", "Escape"
]

searcher = LuceneSearcher("lyric_index")
searcher.set_bm25(k1=0.9, b=0.4)

rows = []

for q in queries:
    hits = searcher.search(q, k=10)

    for rank, hit in enumerate(hits, 1):
        try:
            doc_json = json.loads(hit.lucene_document.get("raw"))
        except:
            doc_json = {}

        rows.append({
            "query": q,
            "rank": rank,
            "song": doc_json.get("song", ""),
            "artist": doc_json.get("artist", ""),
            "text": doc_json.get("original_text", ""),
            "score": hit.score,
            "relevance": ""
        })

df = pd.DataFrame(rows)
df.to_csv("bm25_evaluation_sheet.csv", index=False)
