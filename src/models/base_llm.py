import os
import yaml
import argparse
from huggingface_hub import hf_hub_download
from llama_cpp import Llama


class BaseLLMModel:
    def __init__(self):
        with open("config/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        # Model config
        self.model_repo = self.config["models"]["llm_repo"]
        self.model_filename = self.config["models"]["llm_filename"]
        self.chat_format = self.config["models"]["chat_format"]

        # LLM params
        self.n_ctx = self.config["llm_params"]["n_ctx"]
        self.n_threads = self.config["llm_params"]["n_threads"]
        self.temperature = self.config["llm_params"]["temperature"]
        self.max_tokens = self.config["llm_params"]["max_tokens"]

        self.model_path = None
        self.llm = None

        print("Created LLM instance:")
        print(f"  repo: {self.model_repo}")
        print(f"  file: {self.model_filename}")
        print(f"  ctx: {self.n_ctx}")
        print(f"  threads: {self.n_threads}")

        self._load_model()

        print("_" * 50)

    def _load_model(self):
        # Download or load cached model
        self.model_path = hf_hub_download(
            repo_id=self.model_repo,
            filename=self.model_filename
        )

        self.llm = Llama(
            model_path=self.model_path,
            chat_format=self.chat_format,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False
        )

        print("LLM loaded successfully.")

    def generate(self, query: str, system_prompt: str = None, temperature: float = None):
        messages = []

        if temperature == None:
            temperature = self.temperature

        if system_prompt:
            query = f"{system_prompt}\n\nUser request:{query}"

        messages.append({"role": "user", "content": query})

        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_tokens
        )

        return response["choices"][0]["message"]["content"]

    def rewrite_query(self, query: str) -> str:

        q = (query or "").strip()
        if not q:
            return ""

        system_prompt = """You rewrite movie search queries into a structured format. There are two indexes:
- Plot Index: for story, mood, atmosphere, themes, "movies like X"
- Meta Index: for facts — specific person names (actors/directors), years, genres as filters, ratings, studios

PRIORITY RULES (apply in order):
1. Query mentions a SPECIFIC PERSON'S NAME (actor or director) → ALWAYS use Cast: or Directors:
2. Query mentions a SPECIFIC YEAR or DECADE → use Release_info:
3. Query mentions a SPECIFIC RATING (high rated, Oscar-winning, etc.) → use Rating:
4. Query mentions a GENRE as a hard filter (not as mood/vibe) → use Genres:
5. Everything else (vibes, moods, "like X movie", story themes, atmosphere) → use Plot:

Output formats:
- Plot: "Plot: [natural language query]"
- Meta: "Directors: X" or "Cast: X" or "Genres: X | Release_info: Y" — only include relevant fields, separated by " | "

Examples:
User: "Movies by Christopher Nolan" → Directors: Christopher Nolan
User: "Films with Jeff Goldblum" → Cast: Jeff Goldblum
User: "Jeff Goldblum movies" → Cast: Jeff Goldblum
User: "Leonardo DiCaprio after 2010 with high rating" → Cast: Leonardo DiCaprio | Release_info: after 2010 | Rating: high
User: "Animated films released after 2020" → Genres: Animation | Release_info: after 2020
User: "Sci-fi movies of 2023" → Genres: sci-fi | Release_info: 2023
User: "Something like Interstellar but more emotional" → Plot: emotional space drama like Interstellar
User: "Funny movie like Wedding Crashers" → Plot: comedy like Wedding Crashers
User: "Cult classic like The Room" → Plot: cult classic absurdist like The Room
User: "Dark psychological thriller" → Plot: dark psychological thriller
User: "Good thriller with a twist ending" → Plot: thriller with twist ending

Return ONLY the rewritten query. No explanation, no extra text."""

        try:
            out = self.generate(q, system_prompt, temperature=0.0)
            if not isinstance(out, str):
                return q
            out = out.replace("\r", " ").replace("\n", " ").strip()
            # Normalize repeated spaces
            out = " ".join(out.split())
            # Guardrails
            if not out:
                return q
            if len(out) > 400:
                out = out[:400].rstrip()
            return out
        
        except Exception as e:
            print(f"ERROR while rewriting user query: {e}")
            return q

    def generate_with_context(self, query: str, films: list):

        context_parts = []
        for i, film in enumerate(films, 1):
            title = film.get("title", "Unknown")
            plot = film.get("plot_text", "")
            meta = film.get("meta_text", "")
            context_parts.append(f"Film {i}: {title}\nPlot: {plot}\n{meta}")

        context_text = "\n\n".join(context_parts)
        film_titles = [film["title"] for film in films if film.get("title")]
        titles_list = ", ".join(f'"{t}"' for t in film_titles)

        system_prompt = (
            f"You are a movie recommendation assistant.\n"
            f"You have {len(films)} films from the database that may match the user's request: {titles_list}.\n"
            f"Do NOT recommend films the user has already mentioned in their message.\n\n"
            "Rules:\n"
            "1. First, check the database films below. If 1-2 of them are a good fit, recommend those.\n"
            "2. If the database films are a weak or partial match, recommend the best one from the list AND supplement with 1 film from your own knowledge that fits better.\n"
            "3. If none of the database films are relevant, ignore the list entirely and recommend 1-2 films from your own knowledge.\n"
            "4. Decide how many films to recommend based on the request: recommend 1 film if the request is very specific (exact mood, niche genre, or 'something like X'), recommend 2-3 films if the request is broad or open-ended (e.g. 'good action movies', 'comedies to watch tonight').\n"
            "5. For each recommendation give the exact title and 1-2 sentences on why it fits the request.\n"
            "6. Present all recommendations as a single unified list. Do NOT separate or label films by source (do not write 'from the database', 'from my knowledge', 'I also recommend', etc.).\n"
            "7. Never apologize for results or say you cannot find a match.\n\n"
            f"Database film details:\n{context_text}"
        )

        return self.generate(query, system_prompt=system_prompt, temperature=0.1)


if __name__ == "__main__":
    llm_model = BaseLLMModel()

    parser = argparse.ArgumentParser(description="LLM CLI")
    subparsers = parser.add_subparsers(dest="func", required=True)

    chat_parser = subparsers.add_parser("chat", help="Run interactive chat")

    args = parser.parse_args()

    if args.func == "chat":
        quit = False
        while not quit:
            query = input("Query: ")

            response = llm_model.generate(query)

            print("=" * 65)
            print(response)
            print("=" * 65)

            choice = input("Quit? (y/n): ")
            if choice == "y":
                quit = True