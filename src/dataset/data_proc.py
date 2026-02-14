import os
import yaml
import json
import pandas as pd
import numpy as np
    
class Dataset_proc:
    def __init__(self):
        #FIXME нужно подумать над автоматическим выстроеним пути до файла
        with open("config/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        
        self.MIN_OVERVIEW_WORDS = self.config["data_preprocessing"]["min_overview_words"] #6
        self.MAX_OVERVIEW_WORDS = self.config["data_preprocessing"]["max_overview_words"] #169
    
        #XXX подумать про то что при ините проверять наличие локальных файлов 
    
    def _normalize_title(self, title:str | None):
        if pd.isna(title):
            return ""
        title = str(title).lower().strip()
        title = title.replace("the ", "").replace("a ", "").replace("an ", "")
        title = ''.join(c for c in title if c.isalnum() or c.isspace())
        return ' '.join(title.split())
    
    def _truncate_overview(self, text: str) -> str:
        words = text.split()
        if len(words) > self.MAX_OVERVIEW_WORDS:
            return " ".join(words[:self.MAX_OVERVIEW_WORDS]) + '...'
        return text
    
    #TODO разобраться со скачиванием датасетов
    def download_raw_data(self):
        TMDB_DOWNLOAD_PATH = ...
        LB_DOWNLOAD_PATH = ...
        pass
    
    def preprocess_film_data(self):
        #Init the paths
        TMDB_RAW_PATH = self.config["paths"]["tmdb_raw"]
        LB_RAW_PATH = self.config["paths"]["lb_raw"]
        
        
        #Getting the datasets
        print("Reading the datasets...")
        tmdb = pd.read_csv(TMDB_RAW_PATH, usecols=['title', 'vote_average', 'vote_count', 'popularity', 'release_date', 'status', 'runtime', 'adult', "overview", 'genres', 'production_companies', 'production_countries', 'keywords'])
        letterbox_ = []
        for line in open(LB_RAW_PATH, "r"):
            letterbox_.append(json.loads(line))
        letterbox = pd.DataFrame(letterbox_)
        letterbox.drop(['url', 'reviews', 'poster_url', 'rating'], axis=1, inplace=True)
        
        
        #Merging the datasets
        print("Merging the datasets...")
        letterbox.drop_duplicates(subset=['title'], inplace=True, ignore_index=True)

        tmdb['title_norm'] = tmdb['title'].apply(self._normalize_title)
        letterbox['title_norm'] = letterbox['title'].apply(self._normalize_title)

        tmdb['year'] = pd.to_datetime(tmdb['release_date'], errors='coerce').dt.year.astype('Int64')
        letterbox['year'] = letterbox['year'].astype('Int64')

        #Full outer join on normalized title and year
        merged = pd.merge(
            tmdb,
            letterbox,
            how='outer',
            on=['title_norm', 'year'],
            suffixes=('_tmdb', '_lb')
        )

        #Deduplicating by title_norm
        merged = merged.drop_duplicates(subset=['title_norm'], keep='first')

        #Removing temp columns
        merged = merged.drop(columns=['title_norm'], errors='ignore')
        merged.reset_index(drop=True, inplace=True)
        merged = merged[["title_tmdb", "title_lb", "status", "release_date", "year", "vote_average", "vote_count", "popularity", "runtime", "adult", "overview", "synopsis", "genres_tmdb", "genres_lb", "directors", "cast", "production_countries", "production_companies", "keywords"]]

        
        # del(tmdb)
        # del(letterbox_)
        # del(letterbox)

        
        print("Preprocessing data...")
        #Merging titles
        merged['title'] = merged['title_tmdb'].combine_first(merged['title_lb'])
        merged.drop(columns=['title_tmdb', 'title_lb'], inplace=True)
        
        
        #Merging release_dates
        merged.drop(merged[merged['status'].isin(["Canceled", "Rumored", "Planned"])].index, inplace=True)

        merged['release_info'] = (
            merged['release_date']
            .combine_first(merged['year'])
            .combine_first(merged['status'])
        )
        merged.drop(columns=['release_date', 'year', 'status'], inplace=True)

        
        #Merging overviews
        merged["overview_new"] = merged["overview"].combine_first(merged["synopsis"])
        merged.drop(columns=['overview', 'synopsis'], inplace=True)
        
        
        #Merging genres
        merged["genres"] = merged['genres_tmdb'].combine_first(merged['genres_lb'])
        merged.drop(columns=['genres_tmdb', 'genres_lb'], inplace=True)
        
        
        #Cleaning column names, indexes
        merged = merged[['title', 'release_info', 'overview_new', 'vote_average', 'vote_count', 'popularity', 'runtime', 'genres', 'adult','directors', 'cast', 'production_countries', 'production_companies', 'keywords']]
        merged.rename(columns={'overview_new':'overview'}, inplace=True)
        merged.reset_index(drop=True, inplace=True)
        
        
        #Dropping missing titles
        merged.drop(merged[merged['title'].isin([" ", ""])].index, inplace=True)
        
        
        #Dropping nans and too small/too big overviews
        merged.dropna(subset=['overview'], inplace=True, ignore_index=True)
        
        merged['overview_length'] = merged['overview'].apply(lambda x: len(x.split()))
        merged.drop(merged[merged['overview_length'] < self.MIN_OVERVIEW_WORDS].index, inplace=True)
        merged.reset_index(drop=True, inplace=True)
        
        merged['overview'] = merged['overview'].apply(self._truncate_overview)
        
        merged.drop('overview_length', axis=1, inplace=True)

        
        #Dropping adult titles
        merged.drop(merged[merged['adult'] == True].index, inplace=True)
        merged.drop('adult', axis=1, inplace=True)
        merged.reset_index(drop=True, inplace=True)
        
        #TODO добавить создание текстовых описаний + сохранить        
        
        
if __name__ == "__main__":
    dataset_proc = Dataset_proc()
    dataset_proc.preprocess_film_data()
    #TODO прописать argparser