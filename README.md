# RAG
RAG system for film recommendations

<!-- [![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](https://www.docker.com/) -->

This project develops a RAG system for recommending movies. 

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#inference)
- [Data](#data)
- [Authors](#authors)

## Project Overview


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
- [LetterBox](https://huggingface.co/datasets/pkchwy/letterboxd-all-movie-data) dataset: open-source dataset containing film info.
- [TMDB](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies) dataset: open-source dataset containing film info.
- ...

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

### Pipeline Overview (`dvc.yaml`)
```yaml
stages:
  download  → data/raw/EuroSAT_RGB.zip
  unzip     → data/raw/EuroSAT_RGB/
  prep      → data/prep/EuroSAT_RGB/{image_info.csv, images/}
  cleanup   → (optional) removes raw/zip if configured
```

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

1. To download
```bash
python3 src/dataset/data_proc.py auto
```

### Preprocessing


## Model
<!-- write about embedding model and about base LLM -->

## Authors
- Vasilev Ivan
- Sarantsev Stepan

---