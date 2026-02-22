import os
import yaml, pickle
import faiss
import numpy as np
import pandas as pd
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
    
    def build(self):
        DATA_PREP_PATH = self.config["paths"]["film_data"]
        
        #Check if the preprocessed data is present
        if not os.path.exists(DATA_PREP_PATH):
            print(f"ERROR: No preprocessed data found!\n  Run src/dataset/data_proc.py auto firstly.")
            return
        
        data = pd.read_csv(DATA_PREP_PATH, usecols=["title", "title_plot", "title_meta"])
        data = data.iloc[:300]
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
        index.train(embeddings)
        index.add(embeddings)
        
        #Saving the index
        faiss.write_index(index, self.INDEX_PATH)
        with open(self.config["paths"]["faiss_metadata"], 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Successfully built index!\n  chunks: {index.ntotal} ({index.ntotal//2} plots, {index.ntotal//2} film metadata)\n  saved to {self.INDEX_PATH}")
        
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
    faiss_index.build()
    res = faiss_index.search("A girl called Zero stands in a crowded street, her occupation an assassin.", top_k=10)
    for item in res:
        print(f"{item["chunk_text"]} : {item["similarity"]}")