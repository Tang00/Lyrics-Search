import pandas as pd
import re
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from tqdm import tqdm

tqdm.pandas()

removed_rows = []

def log_removed(reason, df_subset):
    for _, row in df_subset.iterrows():
        removed_rows.append((reason, row.to_dict()))

df = pd.read_csv("spotify_millsongdata.csv")

duplicates = df[df.duplicated(subset=["song", "artist"], keep="first")]
log_removed("Duplicate song+artist", duplicates)

df = df.drop_duplicates(subset=["song", "artist"])

def clean_lyrics(text):
    # Remove section labels like [Chorus], [Verse 1], etc.
    text = re.sub(r"\[.*?\]", " ", text)

    # Remove extra punctuation
    text = re.sub(r"[^a-zA-Z0-9'.,!?;:\s]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Lowercase
    text = text.lower()

    return text

df["clean_lyrics"] = df["text"].progress_apply(clean_lyrics)

df.to_csv("lyrics_cleaned.csv", index=False)

print("Total Songs:", len(df))

print("\nREMOVED ROWS")
for reason, row in removed_rows:
    print(f"\nReason: {reason}")
    print(row)

print("\nTotal removed:", len(removed_rows))
