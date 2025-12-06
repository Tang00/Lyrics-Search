# Lyrics-Search
This project is an exploration in information retrieval techniques through the creation of multiple search models for song lyrics.  
  
The project features 2 models: A BM25 model and an SBERT model. Both are built with Pyserini.  
The models can be interacted with through their respective CLIs.  
  
## Scatterplot
The project also features an interactive scatterplot showing the cosine similarity of each song in the dataset. You can filter by artist to see the similarity of their discography, as well.  
  
This project is hosted on Github Pages and the scatterplot can be accessed here:  
[https://tang00.github.io/Lyrics-Search/](https://tang00.github.io/Lyrics-Search/)  
  
## Dataset
https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset?resource=download
  
## Libraries
This project uses the following libraries:  
 - Pyserini (pyserini)
 - Sentence Transformers (sentence-transformers)
 - UMAP (umap-learn)
 - Plotly (plotly)
 - Pandas (pandas)
 - NumPy (numpy)  
 The package name is written in brackets. To install in command line run:  
```
pip install {package_name}
```  
  
## Running the Project
Unfortunately, the cleaned csv and jsonl files are too large to upload directly to Github. To generate these files:  
```
python clean.py
python csv_to_jsonl.py
```  
These should generate the lyrics_cleaned.csv and collection/pyserini_collection.jsonl necessary to run the CLIs.  
Next, you will need to run this command to create the BM25 index in as the lyric_index directory.  
```
 python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input collection \
  --index lyric_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4
```
The embeddings file sbert_embeddings.npy was uploaded to the project, but if there are any issues, regenerate with:  
```
python embeddings.py
```  
Finally, you can run the CLIs with:  
```
python searcher.py
python sbert_searcher.py
```  
  
## Files
 - clean.py - Dataset cleaning, removing duplicates, removing unecessary text, standardizing white spaces, and lower-case all text
 - csv_to_jsonl.py - Converting cleaned lyrics to jsonl format for building BM25 index with Pyserini
 - searcher.py - CLI for BM25
 - embeddings.py - Generating MiniLM embeddings for SBERT model
 - sbert_searcher.py - CLI for SBERT
 - scatterplot.py - Build interactive Plotly HTML scatterplot of song similarity
 - spotify_millsongdata.csv - original dataset
 - lyrics_cleaned.csv - cleaned lyrics csv (not included, file too large)
 - sbert_embeddings.npy - SBERT embeddings
 - lyrics_embeddings_plot.html - Plotly scatterplot
 - index.html - main html file for hosting on Github Pages, just redirects to the scatterplot
 - collection - used for building BM25 index (not included, file too large)
 - lyric_index - index for BM25 model (not included)
