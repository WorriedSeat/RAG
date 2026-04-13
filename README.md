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
- [Search system](#search-system)
- [LLM](#llm)
- [RAG](#rag-system)
- [Authors](#authors)

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) system for personalized movie recommendations. The system combines data from multiple sources to build a comprehensive movie database, uses Snowflake embedding model and FAISS for efficient retrieval of relevant films based on user queries, and generates natural language responses using a lightweight LLM Llama-3.2-3B-Instruct.

### Why This Project?
<!-- write something cool about the problems that project solves -->

## Features
- **Dual retrieval strategy**: semantic search via FAISS + keyword search via BM25, selected automatically by query type
- **Query rewriting**: meta-queries (director / cast / genre) are restructured by the LLM before retrieval
- **Lightweight local inference**: Llama-3.2-3B-Instruct (Q4_K_M GGUF) — no external API required
- **Web UI + REST API**: Streamlit chat interface FastAPI backend, ready for local or Docker deployment

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
> Requires `indexes/` to be present locally before starting.

### Terminal CLI
> Run the following code from the root of the directory
```bash
python3 src/main.py
```

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
  0.1B model is the smallest one which show great results in the native RAG tasks (Classification: 82.25, Retrieval: 58.41, STS: 76.64) according to [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
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
- [Test-queries](https://github.com/ReDialData/website) dataset: manually preprocessed and sampled 100 real users test queries for RAG-system evaluation.

All datasets cover films up to 2026 announcements.

### Hyperparameters (`config/config.yaml`)
```yaml
data_preprocessing:
  min_overview_words: 15        # film's plot overviews with length less than this are dropped 
  max_overview_words: 169       # film's plot overviews with length more than this are truncated
  top_cast: 10                  # number of top actors from cast which will be present in the description
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
- Cleaning: Handling NaN (e.g., fill year from Letterboxd, drop low-quality overviews <15 words).
- Merging: Exact match on normalized title + year (with fallback for NaN years).
- Filtering: Drop: canceled/rumored/planned status, junk titles, no metadata at all, low-quality films with small description.
- Chunking: Structured text for embeddings (title + plot chunk, metadata chunk with title+ release_info|genres|directors|cast|production companies|keywords|rating).
- Output: Cleaned CSV in prep/ for vector DB indexing.

## Search system
We used FAISS as our vector database. The pipeline of index creation result into search index file `indexes/index_FlatIP_plot.ivf` and metadata file `indexes/faiss_metadata.pkl` for FAISS and `indexes/bm25_meta/` for bm25 search.

### Architecture
We've implemented two separate search systems:
- plot search: search in plots using FAISS
- meta search: search in meta using BM25 

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
> Requires `indexes/` to be present locally before starting.

## LLM
We used [Llama-3.2-3B-Instruct](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) for 2 purposes:
- Rewriting user queries(for better search in plot and in meta)
- Generating film recommendation based on the retrieved results

for this purposes we used system prompts on vanilla LLM with `user_query`/`user_query` & `search_results`  

## RAG system
Orchestrates Search system with vanilla LLM for generating recommendation. Additionally routes the user query either to plot search or to meta search based on embedding of the query. 

## Authors
- Vasilev Ivan
- Sarantsev Stepan

---
