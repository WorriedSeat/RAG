import os, argparse
import torch
import h5py
import faiss
import yaml, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Literal
from sentence_transformers import SentenceTransformer

def _load_config():
    if os.path.exists("config/config.yaml"):
        with open("config/config.yaml", "r") as f:
            return yaml.safe_load(f)
    
    else:
        raise FileNotFoundError("ERROR: config not found at config/config.yaml .\n \
                                ensure you run index.py from project's root directory")


# XXX подумать про добавление доп инфы в метадату
def _create_save_metadata() -> tuple[list, list]:
    config = _load_config()
    DATA_PREP_PATH = config["paths"]["film_data"]
    METADATA_PATH = config["paths"]["faiss_metadata"]
    
    #Check if the preprocessed data is present
    if not os.path.exists(DATA_PREP_PATH):
        raise FileNotFoundError(f"ERROR: No preprocessed data found!\n  Run src/dataset/data_proc.py auto firstly.")
        
    data = pd.read_csv(DATA_PREP_PATH, usecols=["title", "title_plot", "title_meta"])
    #data = data.iloc[:300] #XXX for test_embeddings_build
    data.reset_index(drop=True, inplace=True)

    #Creating text & metadata lists for index
    texts = []
    metadata = []
    for idx, row in data.iterrows():
        #Meta chunk
        texts.append(row["title_meta"])
        metadata.append({"row_idx": idx, "chunk_type": "meta", "title": row["title"], "chunk_text": row["title_meta"]})
        
        #Plot chunk
        texts.append(row["title_plot"])
        metadata.append({"row_idx": idx, "chunk_type": "plot", "title": row["title"], "chunk_text": row["title_plot"]})

    #Saving metadata
    with open(METADATA_PATH, 'wb') as f: 
        pickle.dump(metadata, f)
    print(f"Saved metadata {METADATA_PATH}")

    return texts, metadata

def _create_embeddings(embed_model:SentenceTransformer):
    print("Creating embeddings:")
    
    config = _load_config()
    EMBED_PATH = config["paths"]["embeddings"]
    EMBED_DIM = embed_model.get_sentence_embedding_dimension()
    
    texts, _ = _create_save_metadata()
    
    # Batch encode and append to H5
    batch_size = 400
    with h5py.File(EMBED_PATH, 'w') as hf:
        dset = hf.create_dataset("embeddings", shape=(len(texts), EMBED_DIM), dtype='float32', chunks=True)
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            batch_emb = embed_model.encode(batch, batch_size=batch_size, show_progress_bar=False, precision='float32', normalize_embeddings=True)
            dset[i:i + len(batch_emb)] = batch_emb
            print(f"Batch {i//batch_size + 1} saved")
    
    print(f"Embeddings saved to {EMBED_PATH}")

class FaissIndex:
    def __init__(self):
        config = _load_config()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_model = SentenceTransformer(config["models"]["embedding_model"], device=self.device)
        self.embed_size = self.embed_model.get_sentence_embedding_dimension() #768
        self.max_seq_length = self.embed_model.get_max_seq_length() #512
        self.INDEX_NAME = "indexes/test_index" #config["paths"]["faiss_index"]
        self.EMBED_PATH = config["paths"]["embeddings"]
        self.METADATA_PATH = config["paths"]["faiss_metadata"]
        self.meta_index = None
        self.plot_index = None
        self.metadata = None
        
        print(
            "Created index instance:\n"
            f"  embedding model: {config['models']['embedding_model']} on {self.device}\n"
            f"  embedding size: {self.embed_size}\n"
            f"  max sequence length: {self.max_seq_length}"
        )
        
        self._load()
        
        print("_"*50)

    def _load(self):
        # Reading meta_index
        if os.path.exists(self.INDEX_NAME + "_meta.ivf"):
            self.meta_index = faiss.read_index(self.INDEX_NAME + "_meta.ivf")
            print(f"Loaded meta_index: {self.INDEX_NAME + "_meta.ivf"}")
        else:
            print(f"No meta_index found by path {self.INDEX_NAME + "_meta.ivf"}")
        
        # Reading plot_index
        if os.path.exists(self.INDEX_NAME + "_plot.ivf"):
            self.plot_index = faiss.read_index(self.INDEX_NAME + "_plot.ivf")
            print(f"Loaded plot_index: {self.INDEX_NAME + "_plot.ivf"}")
        else:
            print(f"No plot_index found by path {self.INDEX_NAME + "_plot.ivf"}")
        
        # Reading metadata
        if os.path.exists(self.METADATA_PATH):
            with open(self.METADATA_PATH, "rb") as f:
                self.metadata = pickle.load(f)  
            
            print(f"Loaded metadata: {self.METADATA_PATH}")
        else:
            print(f"No metadata found by path {self.METADATA_PATH}")


    def _get_film_chunk_texts(self, row_idx: int) -> tuple[str, str, str]:
        """
        Returns (title, plot_text, meta_text) for a given film row_idx.

        Assumption: metadata was created in _create_save_metadata() in order:
        meta chunk first, then plot chunk, for each row_idx.
        """
        meta_i = row_idx * 2
        plot_i = row_idx * 2 + 1
        meta = self.metadata[meta_i]
        plot = self.metadata[plot_i]
        
        # Prefer title from meta (same as plot), but keep robust.
        title = meta.get("title") or plot.get("title") or ""
        return title, plot.get("chunk_text", ""), meta.get("chunk_text", "")

    def _build_meta(self, index_path: str | None = None):

        if index_path is None:
            index_path = self.INDEX_NAME + "_meta.ivf"

        # Ensure metadata exists in memory (full chunk list)
        if not os.path.exists(self.METADATA_PATH):
            _, _ = _create_save_metadata()

        # Reload metadata as well (for runtime consistency)
        with open(self.METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)

        print("Loading embeddings (meta only)...")
        with h5py.File(self.EMBED_PATH, "r") as hf:
            emb_ds = hf["embeddings"]
            n_total = emb_ds.shape[0]
            n_meta = (n_total + 1) // 2

            # meta vectors are at even indices: 0,2,4,...
            # Use slicing to avoid loading the full matrix if possible.
            embeddings_meta = emb_ds[::2].astype("float32")

        # Build an ID-mapped index so returned ids correspond to original
        # positions in `self.metadata`.
        base_index = faiss.IndexFlatIP(self.embed_size)
        index = faiss.IndexIDMap2(base_index)

        # Add with original ids (even indices)
        ids = np.arange(0, n_total, 2, dtype=np.int64)
        index.add_with_ids(embeddings_meta, ids)

        print(f"Saving metadata-only index to: {index_path}")
        faiss.write_index(index, index_path)
        print(f"Successfully built meta-only index! vectors: {index.ntotal} (meta chunks)")  
    
    def _build_plot(self, index_path: str | None = None):
        
        if index_path is None:
            index_path = self.INDEX_NAME + "_plot.ivf"

        # Ensure metadata exists in memory (full chunk list)
        if not os.path.exists(self.METADATA_PATH):
            _, _ = _create_save_metadata()
    
        # Reload metadata as well (for runtime consistency)
        with open(self.METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)
    
        print("Loading embeddings (plot only)...")
        with h5py.File(self.EMBED_PATH, "r") as hf:
            emb_ds = hf["embeddings"]
            n_total = emb_ds.shape[0]
            n_plot = (n_total + 1) // 2

            # plot vectors are at odd indices: 1,3,5,...
            # Use slicing to avoid loading the full matrix if possible.
            embeddings_plot = emb_ds[1::2].astype("float32")

        # Build an ID-mapped index so returned ids correspond to original
        # positions in `self.metadata`.
        base_index = faiss.IndexFlatIP(self.embed_size)
        index = faiss.IndexIDMap2(base_index)

        # Add with original ids (odd indices)
        ids = np.arange(1, n_total, 2, dtype=np.int64)
        index.add_with_ids(embeddings_plot, ids)

        print(f"Saving plot-only index to: {index_path}")
        faiss.write_index(index, index_path)
        print(f"Successfully built plot-only index! vectors: {index.ntotal} (plot chunks)")
    
    def build(self):
        if not os.path.exists(self.EMBED_PATH):
            _create_embeddings(self.embed_model)
        
        print("Creating meta index...")
        self._build_meta()
        
        print("Creating plot index...")
        self._build_plot()
    
    def search(self, type:Literal["meta", "plot"], query:str, top_k:int):
        if type == "meta":
            if self.meta_index is None:
                self._load()
            assert self.meta_index != None, "No meta_index file!"
            index = self.meta_index
        
        elif type == "plot":
            if self.plot_index is None:
                self._load()
            assert self.plot_index != None, "No plot_index file!"
            index = self.plot_index
        
        embed_query = self.embed_model.encode([query], precision="float32", normalize_embeddings=True)
        scores, indices = index.search(embed_query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            
            item = self.metadata[idx]
            title, plot_text, meta_text = self._get_film_chunk_texts(item["row_idx"])
            
            results.append({
                "row_idx": item["row_idx"],
                "score": score,
                "title": title,
                "plot_text": plot_text.split("Plot: ")[1],
                "meta_text": " | ".join(meta_text.split(" | ")[1:]),
            })
            
        return results

if __name__ == "__main__":
    #Adding parser for different index functions
    parser = argparse.ArgumentParser(description="Parser for build/search functions")
    func_subparsers = parser.add_subparsers(dest="func", required=True, help="Choose function: build/search")
    
    #Subparser for "build"
    build_parser = func_subparsers.add_parser("build", help="Builds FAISS search index")
    
    #Subparser for "search"
    search_parser = func_subparsers.add_parser("search", help="Run in CLI mode to search for some queries")
    
    args = parser.parse_args()
    
    if args.func == "build":
        faiss_index = FaissIndex()
        faiss_index.build()
    
    elif args.func == "search":
        faiss_index = FaissIndex()
        quit = False
        while not quit:
            type_ = input("Type: ")
            query = input("Query: ")
            res = faiss_index.search(type_, query, top_k=10)
            
            print('='*65)
            for item in res:
                print(f"{item['score']} : {item['title']}\n{item['plot_text']}\n{item['meta_text']}\n")
            print('='*65)
            
            choice = input("Quit?(y/n): ")
            if choice=="y": quit = True