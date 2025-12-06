import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading dataset...")
df = pd.read_csv("lyrics_cleaned.csv")

lyrics = df["clean_lyrics"].astype(str).tolist()

print("Loading SBERT model")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings")
embeddings = model.encode(
    lyrics,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

np.save("sbert_embeddings.npy", embeddings)

print(embeddings.shape)
