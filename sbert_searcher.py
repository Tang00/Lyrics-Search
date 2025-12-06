import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

print("Loading dataset...")
df = pd.read_csv("lyrics_cleaned.csv")

print("Loading embeddings...")
embeddings = np.load("sbert_embeddings.npy")

print("Loading SBERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def sbert_search(query, k=5):
    query_emb = model.encode(query, normalize_embeddings=True)
    scores = util.cos_sim(query_emb, embeddings)[0]

    top_idx = scores.topk(k).indices.tolist()

    results = []
    for i in top_idx:
        results.append({
            "artist": df.loc[i, "artist"],
            "song": df.loc[i, "song"],
            "score": float(scores[i]),
            "lyrics": df.loc[i, "text"]
        })
    return results


def print_results(results):
    for idx, r in enumerate(results, start=1):
        print(f"{idx}. {r['song']} — {r['artist']}")
        print(f"Score: {r['score']:.4f}")
        print(r["lyrics"])


def interactive_cli(k):
    print("\nSBERT Lyric Search CLI")
    print("Type a search query, or type 'quit' to exit.\n")

    while True:
        query = input("Enter query: ").strip()

        if query.lower() in {"quit", "exit"}:
            print("Goodbye")
            break

        if not query:
            continue

        print("\nSearching...\n")

        results = sbert_search(query, k=k)
        print_results(results)


def main():
    parser = argparse.ArgumentParser(description="SBERT Lyric Search CLI")
    parser.add_argument(
        "--k", type=int, default=5, 
        help="Number of search results to return (default: 5)"
    )

    args = parser.parse_args()
    interactive_cli(args.k)


if __name__ == "__main__":
    main()
