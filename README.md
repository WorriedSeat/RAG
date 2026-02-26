# RAG
<!-- [![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](https://www.docker.com/) -->

This project develops a RAG system for recommending movies. 

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Structure](#structure)
- [Inference](#inference)
- [Models](#models)
- [Data](#data)
- [Search Index](#index)
- [Authors](#authors)

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) system for personalized movie recommendations. The system combines data from multiple sources to build a comprehensive movie database, uses embedding model ... for efficient retrieval of relevant films based on user queries, and generates natural language responses using a lightweight LLM ... .

### Why This Project?
<!-- write something cool about the problems that project solves -->

## Features
<!-- write some cool stuff used in project -->

## Structure
```
data/
├── raw/                       # Raw downloaded & extracted data
│   ├── full_dump.jsonl                 # letter-box film dataset
│   ├── TMDBP_movie_dataset_v11.csv     # TMDB film dataset
│   └── ...                             # ... user-queries dataset
│       
│
└── prep/                      # Preprocessed data & search index & test data
    ├── film_data.csv                   # Preprocessed film dataset with descriptions for embedding model
    ├── faiss_index.ivf                 # FAISS search index
    ├── faiss_metadata.pkl              # FAISS metadata
    └── ...                             # Test user queries

```

## Inference
<!-- write about how to try our model locally -->


## Models
In our project we use the following models: 
- **Embedding model**- [Snowflake](https://huggingface.co/Snowflake/snowflake-arctic-embed-m)
  0.1B model is the smallest one which show great results in the native RAG tasks () according to [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- **Base LLM**- 

## Data
1. [Source](#source)
2. [Structure](#structure)
3. [Reproduce](#reproduce)
4. [Preprocessing steps](#preprocessing)

### Source
As a source of our data we used:
- [LetterBox](https://huggingface.co/datasets/pkchwy/letterboxd-all-movie-data) dataset: open-source dataset containing film info, including synopsis, directors, cast, genres, release year.
- [TMDB](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies) dataset: open-source dataset containing film info, including overviews, genres, keywords, release dates, runtime, rating.
- ...

All datasets cover films up to 2026 announcements.

### Hyperparameters (`config/config.yaml`)
```yaml
data_preprocessing:
  min_overview_words: 6         # film's plot overviews with length less than this are dropped 
  max_overview_words: 169       # film's plot overviews with length more than this are truncated
  top_cast: 5                   # number of top actors from cast which will be present in the description
  top_keywords: 10              # number of top keywords to be stored in the description
```
---

### Replicate (`src/dataset/data_proc.py`)
> Run the following code from the root of the directory

1. Automated process based on the presence of data files locally.
```bash
python3 src/dataset/data_proc.py auto
```

2. Manual process `--download` for downloading datasets from datasources, `--prep` for data preprocessing.
```bash
python3 src/dataset/data_proc.py --download --prep
```

### Preprocessing
Preprocessing is handled in `src/dataset/data_proc.py` and includes:
- Downloading raw datasets from Hugging Face and Kaggle.
- Cleaning: Handling NaN (e.g., fill year from Letterboxd, drop low-quality overviews <6 words).
- Merging: Exact match on normalized title + year (with fallback for NaN years).
- Filtering: Drop canceled/rumored/planned status, runtime=0 if low-quality.
- Chunking: Structured text for embeddings (title + plot chunk, metadata chunk with rating/votes/popularity/runtime/cast/keywords).
- Output: Cleaned CSV in prep/ for vector DB indexing.

## Index
We used FAISS as our vector database. The pipeline of index creation result into search index file `data/prep/faiss_index.ivf` and metadata file `data/prep/faiss_metadata.pkl`

### Replicate
> Run the following code from the root of the directory

To replicate the building of the full search index
```bash
python3 src/dataset/index.py build
```
> To see the how index is build on the small part of the dataset you can uncomment the following line in `src/dataset/index.py` and run the same command in terminal
```python
class FaissIndex:
    ...
    def build():
        ...
        # data = data.iloc[:300] #test_build
        ...
```

To test the search results
```bash
python3 src/dataset/index.py search
```
> Works only when `data/prep/faiss_index.ivf` and `data/prep/faiss_metadata.pkl` is present locally!


## Authors
- Vasilev Ivan
- Sarantsev Stepan

---
