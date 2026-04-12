import numpy as np

try:
    from src.dataset.index import FaissIndex
    from src.models.base_llm import BaseLLMModel
except ImportError:
    from dataset.index import FaissIndex
    from models.base_llm import BaseLLMModel

class RAG:
    def __init__(self):
        print(f"Initializing RAG system...\n{'_' * 50}")
        self.index = FaissIndex()
        self.llm = BaseLLMModel()
        self._plot_centroid, self._meta_centroid = self._build_classifier_centroids()
        print("RAG system is ready!!!")

    def _build_classifier_centroids(self):
        plot_queries = [
            "I'm looking for something like Inception with a mind-bending plot",
            "recommend a feel-good romantic comedy for tonight",
            "something funny and lighthearted like Wedding Crashers",
            "scary horror movie with great atmosphere like The Shining",
            "cult classic with absurdist humor like The Room",
            "dark psychological thriller with complex characters",
            "something like Gone Girl with a shocking twist ending",
            "survival thriller set in a remote isolated location",
            "emotional coming of age story about friendship",
            "sci-fi about loneliness and identity crisis",
            "war movie focused on human drama not action",
            "beautiful slow-burn drama with deep storytelling",
            "heist film with clever twists and an unexpected ending",
            "action film with great tense scenes like Mission Impossible",
            "something mysterious and suspenseful",
        ]
        meta_queries = [
            "movies directed by Christopher Nolan",
            "films starring Leonardo DiCaprio released after 2010",
            "best rated sci-fi movies of 2023",
            "Oscar winning films between 2018 and 2022",
            "movies produced by A24 studio",
            "animated films released after 2020",
            "horror films rated R from the last decade",
            "action movies from the 1990s",
            "movies with runtime under 90 minutes",
            "films featuring Meryl Streep with high ratings",
            "movies by director Denis Villeneuve",
            "films starring Jeff Goldblum",
        ]
        em = self.index.embed_model
        plot_vecs = em.encode(plot_queries, normalize_embeddings=True)
        meta_vecs = em.encode(meta_queries, normalize_embeddings=True)
        plot_c = plot_vecs.mean(axis=0); plot_c /= np.linalg.norm(plot_c)
        meta_c = meta_vecs.mean(axis=0); meta_c /= np.linalg.norm(meta_c)
        return plot_c, meta_c

    def classify_query(self, query: str) -> str:
        META_KEYWORDS = (
            "directed by", "starring", "director", "cast:",
            "genre:", "released in", "rated", "produced by",
            "studio", "year:", "runtime", "rating:"
        )
        if any(kw in query.lower() for kw in META_KEYWORDS):
            return "meta"
        vec = self.index.embed_model.encode([query], normalize_embeddings=True)[0]
        return "meta" if np.dot(vec, self._meta_centroid) > np.dot(vec, self._plot_centroid) else "plot"

    @staticmethod
    def _filter_results(results: list, top_k: int = 5, max_unknowns: int = 2) -> list:
        filtered = []
        for r in results:
            meta = r.get("meta_text", "")
            unknown_count = meta.lower().count("unknown")
            if unknown_count <= max_unknowns:
                filtered.append(r)
            if len(filtered) == top_k:
                break
        return filtered if filtered else results[:top_k]

    def process_query(self, query: str):
        search_type = self.classify_query(query)
        print(f"{'_'*20}\nSearch type: {search_type}\n{'_'*20}")

        if search_type == "meta":
            rewritten = self.llm.rewrite_query(query)
            print(f"Rewritten query: {rewritten}")
            search_query = rewritten
        else:
            search_query = query

        candidates = self.index.search(type=search_type, query=search_query, top_k=20)
        results = self._filter_results(candidates, top_k=5)
        print("Search results:")
        for r in results:
            print(r)

        response = self.llm.generate_with_context(query, results)
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
