from dataset.index import FaissIndex
from models.base_llm import BaseLLMModel

class RAG:
    def __init__(self):
        print(f"Initializing RAG system...\n{"_" * 50}")
        self.index = FaissIndex()
        self.llm = BaseLLMModel()
        print("RAG system is ready!!!")
        
    def process_query(self, query: str):
        results = self.index.search(query=query, top_k=5)
        context = [i.get("chunk_text") for i in results]
        response = self.llm.generate_with_context(query, context)
        return response
    
    def terminal_cli(self):
        terminate = False
        print(f"\n\n{"="*50}\nWelcome to RAG film recommendation system!!!\n{"="*50}")
        while not terminate:
            query = input("Search query: ")
            response = self.process_query(query)
            print(f"Response: {response}\n{"_"*15}Answer produced by LLM for reference only{"_"*15}")
            terminate = True if input("Quit(y/n): ").lower() in ("y", "yes") else False
            
if __name__ == "__main__":
    rag = RAG()
    rag.terminal_cli()