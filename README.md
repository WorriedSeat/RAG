# RAG
<!-- [![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](https://www.docker.com/) -->

This project develops a RAG system for recommending movies. 

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#inference)
- [Data](#data)
- [Authors](#authors)

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) system for personalized movie recommendations. The system combines data from multiple sources to build a comprehensive movie database, uses embedding model ... for efficient retrieval of relevant films based on user queries, and generates natural language responses using a lightweight LLM ... .

### Why This Project?
<!-- write something cool about the problems that project solves -->

## Features
<!-- write some cool stuff used in project -->

## Inference
<!-- write about how to try our model locally -->

## Data
1. [Source](#source)
2. [Structure](#structure)
3. [Reproduce](#reproduce)
4. [Preprocessing steps](#preprocessing)

## Source
As a source of our data we used:
- [LetterBox](https://huggingface.co/datasets/pkchwy/letterboxd-all-movie-data) dataset: open-source dataset containing film info, including synopsis, directors, cast, genres, release year.
- [TMDB](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies) dataset: open-source dataset containing film info, including overviews, genres, keywords, release dates, runtime, rating.
- ...

All datasets cover films up to 2026 announcements.

## Structure
```
data/
├── raw/                       # Raw downloaded & extracted data
│   ├── full_dump.jsonl                 # letter-box film dataset
│   ├── TMDBP_movie_dataset_v11.csv     # TMDB film dataset
│   └── ...                             # ... user-queries dataset
│       
│
└── prep/                      # Preprocessed vector db data & test data
    ├── film_data.csv                   # ...
    └── ...                             # ...

```

## Reproduce

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


## Model
<!-- write about embedding model and about base LLM -->

## Authors
- Vasilev Ivan
- Sarantsev Stepan

---