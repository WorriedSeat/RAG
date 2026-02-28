import os
import yaml, json
import kagglehub
import shutil, requests
import argparse
import pandas as pd
    
class Dataset_proc:
    def __init__(self):
        with open("config/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        
        self.MIN_OVERVIEW_WORDS = self.config["data_preprocessing"]["min_overview_words"] #6
        self.MAX_OVERVIEW_WORDS = self.config["data_preprocessing"]["max_overview_words"] #169
        self.TOP_CAST = self.config["data_preprocessing"]["top_cast"]
        self.TOP_KEYWORDS = self.config["data_preprocessing"]["top_keywords"]
        self.TMDB_RAW_PATH = self.config["paths"]["tmdb_raw"]
        self.LB_RAW_PATH = self.config["paths"]["lb_raw"]
        self.FILM_PREP_PATH = self.config["paths"]["film_data"]
        
        
    def _normalize_title(self, title:str | None):
        if pd.isna(title):
            return ""
        title = str(title).lower().strip()
        title = title.replace("the ", "").replace("a ", "").replace("an ", "")
        title = "".join(c for c in title if c.isalnum() or c.isspace())
        return " ".join(title.split())
    
    def _truncate_overview(self, text: str) -> str:
        words = text.split()
        if len(words) > self.MAX_OVERVIEW_WORDS:
            return " ".join(words[:self.MAX_OVERVIEW_WORDS]) + "..."
        return text
    
    def _genres_to_list(self, genres:list | str | float):
        if type(genres) == str:
            genres = genres.split(", ")
        
        return genres
    
    def _process_meta_info(self, row):
        meta_parts = []
        
        #Title
        meta_parts.append(row["title"])
        
        #Release info
        release_info = row["release_info"]
        if pd.isna(release_info):
            meta_parts.append("Release: Unknown")
        elif release_info in ["Released", "In Production", "Post Production"]:
            meta_parts.append(f"{release_info}")
        else:
            meta_parts.append(f"Released in {release_info}")
        
        #Rating, votes, popularity, runtime
        if pd.isna(row["vote_average"]):
            meta_parts.append("Rating: Unknown")
        else:
            meta_parts.append(f"Rating: {row['vote_average']:.1f}/10")
        
        if pd.isna(row["vote_count"]):
            meta_parts.append("Votes: Unknown")
        else:
            meta_parts.append(f"{row['vote_count']:.0f} votes")
        
        if pd.isna(row["popularity"]):
            meta_parts.append("Popularity: Unknown")
        else:
            meta_parts.append(f"Popularity: {row['popularity']:.1f}")
        
        if pd.isna(row["runtime"]) or row["runtime"] == 0:
            meta_parts.append("Runtime: Unknown")
        else:
            meta_parts.append(f"Runtime: {row['runtime']:.0f} min")
        
        #Genres
        genres = row["genres"]
        if type(genres) == float or not genres:
            meta_parts.append("Genres: Unknown")
        else:
            meta_parts.append(f"Genres: {', '.join(genres)}")
            
        #Directors
        directors = row["directors"]
        if type(directors) == float or not directors:
            meta_parts.append("Directors: Unknown")
        else:
            meta_parts.append(f"Directors: {', '.join(directors)}")
        
        #Cast
        cast = row["cast"]
        if type(cast) == float or not cast:
            meta_parts.append("Cast: Unknown")
        else:
            top_cast = ", ".join(cast[:self.TOP_CAST])
            meta_parts.append(f"Cast: {top_cast}")
        
        #Production countries
        if pd.isna(row["production_countries"]):
            meta_parts.append("Production countries: Unknown")
        else:
            meta_parts.append(f"Production countries: {row['production_countries']}")
        
        #Production companies
        if pd.isna(row["production_companies"]):
            meta_parts.append("Production companies: Unknown")
        else:
            meta_parts.append(f"Production companies: {row['production_companies']}")
        
        #Keywords
        if type(row["keywords"]) != float:
            keywords = row["keywords"].split(", ")
            if len(keywords) >= 3:
                kw_str = ", ".join([k for k in keywords[:self.TOP_KEYWORDS]])
                meta_parts.append(f"Tags: {kw_str}")
        
        return " | ".join(meta_parts)
    
    def download_raw_data(self):
        TMDB_DOWNLOAD_PATH = self.config["paths"]["tmdb_download"]
        LB_DOWNLOAD_PATH = self.config["paths"]["lb_download"]
        
        
        os.makedirs(os.path.dirname(self.TMDB_RAW_PATH), exist_ok=True)

        try:
            print("Downloading TMDB dataset...")
            base_data_path = os.path.dirname(self.TMDB_RAW_PATH) #data/raw
            os.environ["KAGGLEHUB_CACHE"] = base_data_path
            tmp_path = kagglehub.dataset_download(TMDB_DOWNLOAD_PATH, force_download=True)
                        
            os.rename(tmp_path+"/"+os.path.basename(self.TMDB_RAW_PATH), self.TMDB_RAW_PATH)
            tmp_folder_name = tmp_path.split("/")[len(base_data_path.split("/"))]
            shutil.rmtree(path=base_data_path + "/" + tmp_folder_name)
            print(f"\tDownloaded in {self.TMDB_RAW_PATH}")
            
        except Exception as e:
            print(f"ERROR (tmdb-download): {e}")
            return
        
        try:
            print("Downloading LetterBox dataset...")
            lb_response = requests.get(LB_DOWNLOAD_PATH, stream=True, timeout=30)
            lb_response.raise_for_status()
            
            with open(self.LB_RAW_PATH, "wb") as f:
                f.write(lb_response.content)
            print(f"\tDownloaded in {self.LB_RAW_PATH}")
        
        except Exception as e:
            print(f"ERROR (lb-download): {e}")
            return
        
        print("All datasets are downloaded!!!")
        
    def preprocess_film_data(self):
               
        #Getting the datasets
        print("Reading the datasets...")
        tmdb = pd.read_csv(self.TMDB_RAW_PATH, usecols=["title", "vote_average", "vote_count", "popularity", "release_date", "status", "runtime", "adult", "overview", "genres", "production_companies", "production_countries", "keywords"])
        letterbox_ = []
        for line in open(self.LB_RAW_PATH, "r"):
            letterbox_.append(json.loads(line))
        letterbox = pd.DataFrame(letterbox_)
        letterbox.drop(["url", "reviews", "poster_url", "rating"], axis=1, inplace=True)
        
        
        #Merging the datasets
        print("Merging the datasets...")
        letterbox.drop_duplicates(subset=["title"], inplace=True, ignore_index=True)

        tmdb["title_norm"] = tmdb["title"].apply(self._normalize_title)
        letterbox["title_norm"] = letterbox["title"].apply(self._normalize_title)

        tmdb["year"] = pd.to_datetime(tmdb["release_date"], errors="coerce").dt.year.astype("Int64")
        letterbox["year"] = letterbox["year"].astype("Int64")

        #Full outer join on normalized title and year
        merged = pd.merge(
            tmdb,
            letterbox,
            how="outer",
            on=["title_norm", "year"],
            suffixes=("_tmdb", "_lb")
        )

        #Deduplicating by title_norm
        merged = merged.drop_duplicates(subset=["title_norm"], keep="first")

        #Removing temp columns
        merged = merged.drop(columns=["title_norm"], errors="ignore")
        merged.reset_index(drop=True, inplace=True)
        merged = merged[["title_tmdb", "title_lb", "status", "release_date", "year", "vote_average", "vote_count", "popularity", "runtime", "adult", "overview", "synopsis", "genres_tmdb", "genres_lb", "directors", "cast", "production_countries", "production_companies", "keywords"]]

        
        del(tmdb)
        del(letterbox_)
        del(letterbox)

        
        print("Preprocessing data...")
        #Merging titles
        merged["title"] = merged["title_tmdb"].combine_first(merged["title_lb"])
        merged.drop(columns=["title_tmdb", "title_lb"], inplace=True)
        
        
        #Merging release_dates
        merged.drop(merged[merged["status"].isin(["Canceled", "Rumored", "Planned"])].index, inplace=True)

        merged["release_info"] = (
            merged["release_date"]
            .combine_first(merged["year"])
            .combine_first(merged["status"])
        )
        merged.drop(columns=["release_date", "year", "status"], inplace=True)

        
        #Merging overviews
        merged["overview_new"] = merged["overview"].combine_first(merged["synopsis"])
        merged.drop(columns=["overview", "synopsis"], inplace=True)
        
        
        #Merging genres
        merged["genres"] = merged["genres_tmdb"].combine_first(merged["genres_lb"])
        merged.drop(columns=["genres_tmdb", "genres_lb"], inplace=True)
        merged["genres"] = merged["genres"].apply(self._genres_to_list)
        
        
        #Cleaning column names, indexes
        merged = merged[["title", "release_info", "overview_new", "vote_average", "vote_count", "popularity", "runtime", "genres", "adult","directors", "cast", "production_countries", "production_companies", "keywords"]]
        merged.rename(columns={"overview_new":"overview"}, inplace=True)
        merged.reset_index(drop=True, inplace=True)
        
        
        #Dropping missing titles
        merged.drop(merged[merged["title"].isin([" ", ""])].index, inplace=True)
        
        
        #Dropping nans and too small/too big overviews
        merged.dropna(subset=["overview"], inplace=True, ignore_index=True)
        
        merged["overview_length"] = merged["overview"].apply(lambda x: len(x.split()))
        merged.drop(merged[merged["overview_length"] < self.MIN_OVERVIEW_WORDS].index, inplace=True)
        merged.reset_index(drop=True, inplace=True)
        
        merged["overview"] = merged["overview"].apply(self._truncate_overview)
        
        merged.drop("overview_length", axis=1, inplace=True)

        
        #Dropping adult titles
        merged.drop(merged[merged["adult"] == True].index, inplace=True)
        merged.drop("adult", axis=1, inplace=True)
        merged.reset_index(drop=True, inplace=True)
        
        
        #Creating descriptions for embeddings
        print("Creating descriptions...")
        merged["title_plot"] = merged.apply(lambda row: f"Title: {row["title"]} Plot: {row["overview"]}", axis=1)        
        merged["title_meta"] = merged.apply(self._process_meta_info, axis=1)

        
        #Saving preprocessed dataset
        os.makedirs(os.path.dirname(self.FILM_PREP_PATH), exist_ok=True)
        merged.to_csv(self.FILM_PREP_PATH)
        print(f"Preprocessed dataset saved to: {self.FILM_PREP_PATH}")
        
if __name__ == "__main__":
    dataset_proc = Dataset_proc()
    
    parser = argparse.ArgumentParser(description="Parser for auto or manual mode")

    #Subparsers for main modes
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Choose mode: auto or man")

    #Subparser for "auto"
    auto_parser = subparsers.add_parser("auto", help="Automatic mode: checks files in directory and performs actions")

    #Subparser for "man"
    man_parser = subparsers.add_parser("man", help="Manual mode: specify actions with flags")
    man_parser.add_argument("--download", action="store_true", help="Perform download of datasets")
    man_parser.add_argument("--prep", action="store_true", help="Perform prep action")

    args = parser.parse_args()

    if args.mode == "auto":
        print("Running in auto mode")
        commands = {
            "download": [True, dataset_proc.download_raw_data],
            "preprocess": [True, dataset_proc.preprocess_film_data]
        }
        
        
        if os.path.exists(dataset_proc.TMDB_RAW_PATH) and os.path.exists(dataset_proc.LB_RAW_PATH):
            commands["download"][0] = False

        if os.path.exists(dataset_proc.FILM_PREP_PATH):
            commands["preprocess"][0] = False

            
        for command in commands.keys():
            if commands.get(command)[0]:
                print(f"  Executing {command} step:")
                commands.get(command)[1]()
        
    elif args.mode == "man":
        print("Running in manual mode")
        
        if args.download:
            print("Downloading data:")
            dataset_proc.download_raw_data()
            
        if args.prep:
            print("Preprocessing data:")
            dataset_proc.preprocess_film_data()