import os
import yaml, pickle
import faiss
import numpy as np
import pandas as pd
import argparse
from sentence_transformers import SentenceTransformer

class FaissIndex:
    def __init__(self):
        with open("config/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
                
        self.embed_model = SentenceTransformer(self.config["models"]["embedding_model"])
        self.embed_size = self.embed_model.get_sentence_embedding_dimension() #768
        self.max_seq_length = self.embed_model.get_max_seq_length() #512
        self.INDEX_PATH = self.config["paths"]["faiss_index"]
        self.index = None
        self.metadata = None
        
        print(f"Created index instance:\n  embedding model: {self.config["models"]["embedding_model"]}\n  embedding size: {self.embed_size}\n  max sequence length: {self.max_seq_length}")
        
        #Reading index & metadata if created
        if os.path.exists(self.INDEX_PATH):
            self.index = faiss.read_index(self.INDEX_PATH)
            
            with open(self.config["paths"]["faiss_metadata"], "rb") as f:
                self.metadata = pickle.load(f)   
            
            print(f"Found and loaded:\n  index: {self.INDEX_PATH}\n  metadata: {self.config["paths"]["faiss_metadata"]}")
        
        print("_"*50)
    
    def build(self, gpu_enabled=True):
        DATA_PREP_PATH = self.config["paths"]["film_data"]
        print("Building FAISS search index")
        
        #Check if the preprocessed data is present
        if not os.path.exists(DATA_PREP_PATH):
            print(f"ERROR: No preprocessed data found!\n  Run src/dataset/data_proc.py auto firstly.")
            return
        
        data = pd.read_csv(DATA_PREP_PATH, usecols=["title", "title_plot", "title_meta"])
        # data = data.iloc[:300] #test_build
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
        
        #Creating an embeddings
        print("Creating embeddings:")
        embeddings = self.embed_model.encode(texts, batch_size=128, show_progress_bar=True, precision='float32', normalize_embeddings=True)
        
        #Creating an index
        num_clusters = int(np.sqrt(embeddings.shape[0])) #XXX hyperparam to play with
        quantizer = faiss.IndexFlatL2(self.embed_size)
        index = faiss.IndexIVFFlat(quantizer, self.embed_size, num_clusters)
        
        #Training & adding & saving gpu/cpu index
        if gpu_enabled:
            try:
                print("Moving index to GPU...")
                resources = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(resources, 0, index)
                
                print("Training index on GPU...")
                gpu_index.train(embeddings)
                
                print("Adding embeddings on GPU...")
                gpu_index.add(embeddings)
                
                print("Saving CPU-version of index...")
                cpu_index = faiss.index_gpu_to_cpu(gpu_index)
                faiss.write_index(cpu_index, self.INDEX_PATH)        
                print(f"Successfully built index!\n  chunks: {cpu_index.ntotal} ({cpu_index.ntotal//2} plots, {cpu_index.ntotal//2} film metadata)\n  saved to {self.INDEX_PATH}")
            
            except Exception as e:
                print(f"GPU indexing failed: {e}")
                print("Falling back to CPU")
                gpu_enabled = False
            
        if not gpu_enabled:
            
            print("Training index on CPU...") 
            index.train(embeddings)
            
            print("Adding Embeddings on CPU...")
            index.add(embeddings)
            faiss.write_index(index, self.INDEX_PATH)
            print(f"Successfully built index!\n  chunks: {index.ntotal} ({index.ntotal//2} plots, {index.ntotal//2} film metadata)\n  saved to {self.INDEX_PATH}")
        
        #Saving metadata
        with open(self.config["paths"]["faiss_metadata"], 'wb') as f:
            pickle.dump(metadata, f)
        
        
    def search(self, query:str, top_k:int):
        #Checking if index accessible
        if self.index == None:
            if os.path.exists(self.INDEX_PATH):
                self.index = faiss.read_index(self.INDEX_PATH)
                with open(self.config["paths"]["faiss_metadata"], "rb") as f:
                    self.metadata = pickle.load(f) 
            else:
                raise ValueError(f"ERROR: No faiss index found!")     
        
        embed_query = self.embed_model.encode([query], prompt_name="query", precision="float32", normalize_embeddings=True)
        self.index.nprobe = 10 #XXX hyperparam to play with
        
        distances, indices = self.index.search(embed_query, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: continue
            
            results.append({
                "chunk_text": self.metadata[idx]["chunk_text"],
                "similarity": dist
            })
        
        #TODO подумать по поводу постфильтеринга
        
        return results


if __name__ == "__main__":
    faiss_index = FaissIndex()
    
    #Adding parser for different index functions
    parser = argparse.ArgumentParser(description="Parser for build/search functions")
    func_subparsers = parser.add_subparsers(dest="func", required=True, help="Choose function: build/search")
    
    #Subparser for "build"
    build_parser = func_subparsers.add_parser("build", help="Builds FAISS search index")
    
    #Subparser for "search"
    search_parser = func_subparsers.add_parser("search", help="Run in CLI mode to search for some queries")
    
    args = parser.parse_args()
    
    if args.func == "build":
        faiss_index.build()
    
    elif args.func == "search":
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