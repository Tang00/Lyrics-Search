from pyserini.search.lucene import LuceneSearcher
import json

def run_bm25_cli(index_dir="lyric_index", top_k=5):

    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(k1=0.9, b=0.4)

    print("\nWelcome to Lyrics Search CLI! Type 'exit' to quit.\n")

    while True:
        query = input("Enter your search query: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        if len(query) == 0:
            continue

        # Search
        hits = searcher.search(query, k=top_k)
        if len(hits) == 0:
            print("No results found.\n")
            continue

        print(f"\nTop {top_k} results for '{query}':\n")

        for i, hit in enumerate(hits, 1):
            doc_json = json.loads(hit.lucene_document.get("raw"))
            artist = doc_json.get("artist", "")
            song = doc_json.get("song", "")
            lyrics = doc_json.get("original_text", "")
            score = hit.score

            print(f"{i}. {artist} - {song} (score: {score:.2f})")
            print(f"   Lyrics: {lyrics}\n")


if __name__ == "__main__":
    run_bm25_cli(index_dir="lyric_index", top_k=5)