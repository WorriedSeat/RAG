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
        
        print(f"Created index instance:\n  embedding model: {config["models"]["embedding_model"]} on {self.device}\n  embedding size: {self.embed_size}\n  max sequence length: {self.max_seq_length}")
        
        #Reading index & metadata if created
        if os.path.exists(self.INDEX_PATH):
            self.index = faiss.read_index(self.INDEX_PATH)
            
            with open(config["paths"]["faiss_metadata"], "rb") as f:
                self.metadata = pickle.load(f)   
            
            print(f"Found and loaded:\n  index: {self.INDEX_PATH}\n  metadata: {config["paths"]["faiss_metadata"]}")
        else:
            print(f"No index/metadata file found.\n \
                Check paths or build index")
            
        print("_"*50)

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
            _create_save_metadata()
        
        
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

        
    def search(self, query:str, top_k:int):   
        embed_query = self.embed_model.encode([query], precision="float32", normalize_embeddings=True)
        # self.index.nprobe = 10 #XXX hyperparam to play with
        
        scores, indices = self.index.search(embed_query, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            
            results.append({
                "chunk_text": self.metadata[idx]["chunk_text"],
                "similarity": float(score)
            })
        
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
                print(f"{item["similarity"]} : {item["chunk_text"]}\n")
            print('='*65)
            
            choice = input("Quit?(y/n): ")
            if choice=="y": quit = True 