import pandas as pd
import json

df = pd.read_csv("lyrics_cleaned.csv")

with open("collection/pyserini_collection.jsonl", "w") as f:
    for i, row in df.iterrows():
        doc = {
            "id": str(i),
            "contents": row["clean_lyrics"],
            "original_text": row["text"],
            "artist": row["artist"],
            "song": row["song"],
            "link": row["link"]
        }
        f.write(json.dumps(doc) + "\n")