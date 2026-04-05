from dataset.index import FaissIndex
from models.base_llm import BaseLLMModel

class RAG:
    def __init__(self):
        print(f"Initializing RAG system...\n{'_' * 50}")
        self.index = FaissIndex()
        self.llm = BaseLLMModel()
        print("RAG system is ready!!!")
    
    def route_query(self, llm_response: str):
        text = llm_response.strip().lower()
                
        if text.startswith("plot:"):
            return "plot"
        
        meta_prefixes = [
            "directors:", "director:", 
            "cast:", 
            "genres:", "genre:", 
            "rating:", 
            "release_info:", "release:", "year:",
            "keywords:", "keyword:",
            "production:", "studio:"
        ]

        for prefix in meta_prefixes:
            if text.startswith(prefix):
                return "meta"

        for prefix in meta_prefixes:
            if prefix in text[:35]:
                return "meta"
        
        else:
            return "plot"
    
    def process_query(self, query: str):
        rewritten_query = self.llm.rewrite_query(query)
        print(f"{'_'*20}\nrewritten query: {rewritten_query}\n{'_'*20}")
        
        search_type = self.route_query(rewritten_query)
        print(f"Search type: {search_type}")      
        
        results = self.index.search(type=search_type, query=rewritten_query, top_k=5)
        print("search results:")
        for i in results:
            print(i)
        
        context = []
        for film in results:
            context.append(film.get("plot_text", ""))
            context.append(film.get("meta_text", ""))
        response = self.llm.generate_with_context(query, context)
        return response
    
    def terminal_cli(self):
        terminate = False
        print(f"\n\n{'='*50}\nWelcome to RAG film recommendation system!!!\n{'='*50}")
        while not terminate:
            query = input("Search query: ")
            response = self.process_query(query)
            print(f"Response: {response}\n{'_'*15}Answer produced by LLM for reference only{'_'*15}")
            terminate = True if input("Quit(y/n): ").lower() in ("y", "yes") else False
            
if __name__ == "__main__":
    rag = RAG()
    rag.terminal_cli()