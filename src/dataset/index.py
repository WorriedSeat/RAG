import os, argparse
import torch
import h5py
import faiss
import yaml, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def _load_config():
    if os.path.exists("config/config.yaml"):
        with open("config/config.yaml", "r") as f:
            return yaml.safe_load(f)
    
    else:
        raise FileNotFoundError("ERROR: config not found at config/config.yaml .\n \
                                ensure you run index.py from project's root directory")

def _create_save_metadata() -> tuple[list, list]:
    config = _load_config()
    DATA_PREP_PATH = config["paths"]["film_data"]
    METADATA_PATH = config["paths"]["faiss_metadata"]
    
    #Check if the preprocessed data is present
    if not os.path.exists(DATA_PREP_PATH):
        raise FileNotFoundError(f"ERROR: No preprocessed data found!\n  Run src/dataset/data_proc.py auto firstly.")
        
    data = pd.read_csv(DATA_PREP_PATH, usecols=["title", "title_plot", "title_meta"])
    # data = data.iloc[:300] #XXX for test_embeddings_build
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
    batch_size = 128
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
        self.INDEX_PATH = config["paths"]["faiss_index"]
        self.EMBED_PATH = config["paths"]["embeddings"]
        self.METADATA_PATH = config["paths"]["faiss_metadata"]
        self.index = None
        self.metadata = None
        
        print(
            "Created index instance:\n"
            f"  embedding model: {config['models']['embedding_model']} on {self.device}\n"
            f"  embedding size: {self.embed_size}\n"
            f"  max sequence length: {self.max_seq_length}"
        )
        
        #Reading index & metadata if created
        if os.path.exists(self.INDEX_PATH):
            self.index = faiss.read_index(self.INDEX_PATH)
            
            with open(config["paths"]["faiss_metadata"], "rb") as f:
                self.metadata = pickle.load(f)   
            
            print(
                "Found and loaded:\n"
                f"  index: {self.INDEX_PATH}\n"
                f"  metadata: {config['paths']['faiss_metadata']}"
            )
        else:
            print(f"No index/metadata file found.\n \
                Check paths or build index")
            
        print("_"*50)

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


    def build(self):
        #Loading embeddings
        print("Loading embeddings...")
        try:
            with h5py.File(self.EMBED_PATH, 'r') as hf:
                embeddings = hf['embeddings'][:].astype('float32') 
        
        except FileNotFoundError:
            choice = input(f"No embeddings found {self.EMBED_PATH}. Ensure that file is present.\n \
                Create embeddings? (y/n): ")
            
            if choice.lower() == 'n':
                print("Can't create index without embeddings")
                return 
            
            else:
                _create_embeddings(self.embed_model)
                with h5py.File(self.EMBED_PATH, 'r') as hf:
                    embeddings = hf['embeddings'][:].astype('float32')
        
        if not os.path.exists(self.METADATA_PATH):
            _, _ = _create_save_metadata()
        
        
        #Creating an index
        # num_clusters = int(np.sqrt(self.embed_size)) #XXX hyperparam to play with
        # quantizer = faiss.IndexFlatL2(self.embed_size)
        # index = faiss.IndexIVFFlat(quantizer, self.embed_size, num_clusters)
        
        index = faiss.IndexFlatIP(self.embed_size)
        
        #Training & adding & saving gpu/cpu index
        if torch.cuda.is_available():
            print("Moving index to GPU...")
            resources = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(resources, 0, index)
            
            # print("Training index on GPU...")
            # gpu_index.train(embeddings)
            
            print("Adding embeddings on GPU...")
            gpu_index.add(embeddings)
            
            index = faiss.index_gpu_to_cpu(gpu_index)
            
        else:
            # print("Training index on CPU...") 
            # index.train(embeddings)
            
            print("Adding Embeddings on CPU...")
            index.add(embeddings)
                   
        print("Saving CPU-version of index...")
        faiss.write_index(index, self.INDEX_PATH)        
        print(f"Successfully built index!\n \
            chunks: {index.ntotal} ({index.ntotal//2} plots, {index.ntotal//2} film metadata)\n  saved to {self.INDEX_PATH}")

        
    def search(self, query: str, top_k: int, top_k_chunks_multiplier: int = 30, return_debug_chunks: bool = False):
        """
        Search FAISS by query and return top_k FILMS (aggregated by row_idx), not chunks.

        Returns list[dict] with keys:
        - row_idx, title, film_score, plot_text, meta_text
        - optionally debug_chunks (if return_debug_chunks=True)
        """
        if self.index is None or self.metadata is None:
            raise RuntimeError("Index/metadata is not loaded. Build the index first or check paths.")

        embed_query = self.embed_model.encode([query], precision="float32", normalize_embeddings=True)
        # self.index.nprobe = 10 #XXX hyperparam to play with
        
        top_k_chunks = max(top_k, int(top_k) * int(top_k_chunks_multiplier))
        scores, indices = self.index.search(embed_query, top_k_chunks)
        
        by_film = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            m = self.metadata[idx]
            row_idx = m["row_idx"]
            chunk_type = m.get("chunk_type")
            s = float(score)

            agg = by_film.get(row_idx)
            if agg is None:
                agg = {
                    "row_idx": int(row_idx),
                    "title": m.get("title", ""),
                    "film_score": s,  # max over chunks (initialized)
                    "plot_best": -1.0,
                    "meta_best": -1.0,
                }
                if return_debug_chunks:
                    agg["debug_chunks"] = []
                by_film[row_idx] = agg
            else:
                if s > agg["film_score"]:
                    agg["film_score"] = s


            if chunk_type == "plot" and s > agg["plot_best"]:
                agg["plot_best"] = s
            elif chunk_type == "meta" and s > agg["meta_best"]:
                agg["meta_best"] = s

            if return_debug_chunks:
                agg["debug_chunks"].append(
                    {"chunk_type": chunk_type, "similarity": s, "chunk_text": m.get("chunk_text", "")}
                )

        # Sort films by aggregated score
        ranked = sorted(by_film.values(), key=lambda x: x["film_score"], reverse=True)[:top_k]

        # Always attach BOTH chunks for each film
        results = []
        for item in ranked:
            title, plot_text, meta_text = self._get_film_chunk_texts(item["row_idx"])
            out = {
                "row_idx": item["row_idx"],
                "title": title,
                "film_score": float(item["film_score"]),
                "plot_text": plot_text,
                "meta_text": meta_text,
            }
            if return_debug_chunks:
                out["debug_chunks"] = item.get("debug_chunks", [])
                out["plot_best"] = float(item["plot_best"])
                out["meta_best"] = float(item["meta_best"])
            results.append(out)
        
        #TODO подумать по поводу постфильтеринга
        
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
            query = input("Query: ")
            res = faiss_index.search(query, top_k=10)
            
            print('='*65)
            for item in res:
                print(f"{item['film_score']} : {item['title']}\n{item['plot_text']}\n{item['meta_text']}\n")
            print('='*65)
            
            choice = input("Quit?(y/n): ")
            if choice=="y": quit = True