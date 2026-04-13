# RAG
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](https://www.docker.com/)

This project develops a RAG system for recommending movies. 

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Structure](#structure)
- [Inference](#inference)
- [Docker](#docker)
- [Models](#models)
- [Data](#data)
- [Search Index](#index)
- [Authors](#authors)

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) system for personalized movie recommendations. The system combines data from multiple sources to build a comprehensive movie database, uses Snowflake embedding model and FAISS for efficient retrieval of relevant films based on user queries, and generates natural language responses using a lightweight LLM Llama-3.2-3B-Instruct.

### Why This Project?
<!-- write something cool about the problems that project solves -->

## Features
- **Dual retrieval strategy**: semantic search via FAISS + keyword search via BM25, selected automatically by query type
- **Query rewriting**: meta-queries (director / cast / genre) are restructured by the LLM before retrieval
- **Lightweight local inference**: Llama-3.2-3B-Instruct (Q4_K_M GGUF) — no external API required
- **Web UI + REST API**: Streamlit chat interface backed by FastAPI, ready for local or Docker deployment

## Structure
```
RAG/
├── config/
│   └── config.yaml                     # All hyperparameters and paths
├── data/
│   ├── raw/                            # Raw downloaded datasets (gitignored)
│   │   ├── full_dump.jsonl             # Letterboxd dataset
│   │   └── TMDB_movie_dataset_v11.csv  # TMDB dataset
│   └── prep/                           # Preprocessed data (gitignored)
│       ├── film_data.csv               # Cleaned & merged film records
│       └── embeddings_full_snowflake.npy
├── indexes/                            # Search indexes (gitignored)
│   ├── index_FlatIP_plot.ivf           # FAISS plot index
│   ├── faiss_metadata.pkl
│   └── bm25_meta/                      # BM25 metadata index
├── src/
│   ├── dataset/
│   │   ├── data_proc.py                # Download & preprocess datasets
│   │   └── index.py                    # Build FAISS + BM25 indexes
│   ├── deployment/
│   │   ├── api/api.py                  # FastAPI backend
│   │   └── app/app.py                  # Streamlit frontend
│   ├── models/
│   │   └── base_llm.py                 # Llama-3.2 wrapper
│   └── main.py                         # RAG orchestration
├── eval.py                             # Evaluation script
├── Dockerfile
├── docker-compose.yaml
└── requirements.txt
```

## Inference

### Terminal CLI
> Run the following code from the root of the directory
```bash
python3 src/main.py
```
Note that this code executes only when all necessary files are present

### Inference app
Run the FastAPI backend and Streamlit UI separately:

```bash
# Terminal 1 — API server (from project root)
uvicorn src.deployment.api.api:app --host 0.0.0.0 --port 8000

# Terminal 2 — Streamlit UI
streamlit run src/deployment/app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Docker

> Requires `indexes/` and `data/prep/` to be present locally before starting.

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI (REST) | http://localhost:8000 |
| Health check | http://localhost:8000/health |

The first startup downloads the embedding model and Llama-3.2 GGUF (~2 GB total) into a named Docker volume (`hf_cache`). Subsequent starts reuse the cache.

**Rebuild after code changes:**
```bash
docker compose up --build
```

**Run only the API (no UI):**
```bash
docker compose up api
```

## Models
In our project we use the following models: 
- **Embedding model**- [Snowflake](https://huggingface.co/Snowflake/snowflake-arctic-embed-m)
  0.1B model is the smallest one which show great results in the native RAG tasks () according to [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- **Base LLM** — [Llama-3.2-3B-Instruct](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) (Q4_K_M GGUF quantization via `llama-cpp-python`). Runs fully on CPU, no GPU required.

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
