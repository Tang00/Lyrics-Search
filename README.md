# Lyrics-Search
This project is an exploration in information retrieval techniques through the creation of multiple search models for song lyrics.  
The project features 2 models: A BM25 model and an SBERT model. Both are built with Pyserini. 
The models can be interacted with through their respective CLI's in searcher.py (BM25) and sbert_searcher.py (SBERT).  
The project also features an interactive scatterplot showing the cosine similarity of each song in the dataset.  

## Dataset
https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset?resource=download
  
## Libraries
This project uses the following libraries:  
 - Pyserini (pyserini)
 - Sentence Transformers (sentence-transformers)(required)
 - UMAP (umap-learn)
 - Plotly (plotly)
 - Pandas (pandas) (required)
 - NumPy (numpy) (required)  
To run the CLIs only those marked with required are necessary. The other libraries were used to build the model index/embeddings. The package name is written in brackets. To install in command line run:  
```
pip install {package_name}
```  
  
## Running the CLIs
```
python3 searcher.py
python3 sbert_searcher.py
```  
  
## Files
 - clean.py - Dataset cleaning, removing duplicates, removing unecessary text, standardizing white spaces, and lower-case all text
 - csv_to_jsonl.py - Converting cleaned lyrics to jsonl format for building BM25 index with Pyserini
 - searcher.py - CLI for BM25
 - embeddings.py - Generating MiniLM embeddings for SBERT model
 - sbert_searcher.py - CLI for SBERT
 - scatterplot.py - Build interactive Plotly HTML scatterplot of song similarity
 - spotify_millsongdata.csv - original dataset
 - lyrics_cleaned.csv - cleaned lyrics csv
 - sbert_embeddings.npy - SBERT embeddings
 - lyrics_embeddings_plot.html - Plotly scatterplot
 - collection - used for building BM25 index
 - lyric_index - index for BM25 model
  
## Command Line
The following command was run in order to create the BM25 index:  
 ```
 python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input collection \
  --index lyric_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4
  ```