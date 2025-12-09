import pandas as pd
import numpy as np
import umap
import plotly.express as px

def plot_embeddings(
    csv_path="data/lyrics_cleaned.csv",
    embeddings_path="sbert_embeddings.npy",
    output_html="lyrics_embeddings_plot.html"
):
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    print("Loading embeddings...")
    embeddings = np.load(embeddings_path)

    if len(df) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch: CSV has {len(df)} rows, embeddings have {embeddings.shape[0]} rows"
        )

    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(
        n_neighbors=50,
        min_dist=0.1,
        spread=5.0,
        metric="cosine",
        random_state=42
    )
    points_2d = reducer.fit_transform(embeddings)

    df["x"] = points_2d[:, 0]
    df["y"] = points_2d[:, 1]

    x_range = [df["x"].min() - 0.1, df["x"].max() + 0.1]
    y_range = [df["y"].min() - 0.1, df["y"].max() + 0.1]

    fig = px.scatter(
        df,
        x="x",
        y="y",
        hover_name="song",
        hover_data={"artist": True, "text": False},
        color="artist",
        title="UMAP Visualization of Song Embeddings (SBERT)",
        opacity=0.7
    )

    fig.update_layout(
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range),
        legend=dict(
            title="Artist",
            itemsizing="constant",
            itemclick="toggleothers",
            itemdoubleclick="toggle"
        )
    )

    print(f"Saving HTML to {output_html}...")
    fig.write_html(output_html, auto_open=True)

if __name__ == "__main__":
    plot_embeddings()
