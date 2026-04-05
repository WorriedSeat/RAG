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

        system_prompt = """
        You are an intelligent query router for a movie recommendation RAG system that has two separate search indexes:

        1. **Plot Index** – searches by movie plot, story, atmosphere, mood, and narrative elements.
        2. **Meta Index** – searches by factual metadata (director, cast, year, rating, genres, keywords, production companies, etc.).

        Your job:
        - Carefully analyze the user's query.
        - Decide whether the Plot Index or Meta Index is more appropriate.
        - Rewrite the query in the correct format.

        Strict rules:
        - If the query is about **plot, story, atmosphere, mood, "movies like X", "in the style of X", genre feeling, or narrative** → use **Plot Index**. Start your response with "Plot:".
        - If the query is about **specific facts** (director, actors, year, rating, genres, keywords, studio, runtime, etc.) → use **Meta Index**. Use structured prefixes.
        - Only include metadata prefixes that are **explicitly or strongly implied** in the query. Do not add fields that were not mentioned.
        - Respond with **only** the rewritten query. No explanations, no extra text, no apologies.

        Output formats:

        - Plot Index: "Plot: [rewritten natural language query]"
        - Meta Index: "Directors: X | Cast: Y | Genres: Z | Rating: high | Release_info: after 2020 | Keywords: ..." (only relevant fields, separated by " | ")

        Few-shot examples:

        User: "Movies by Christopher Nolan"
        → "Directors: Christopher Nolan"

        User: "Something like Interstellar but with time travel"
        → "Plot: time travel like Interstellar"

        User: "Leonardo DiCaprio movies after 2010 with high rating"
        → "Cast: Leonardo DiCaprio | Release_info: after 2010 | Rating: high"

        User: "Dark psychological thrillers with revenge plot"
        → "Plot: dark psychological thriller revenge"

        User: "Best sci-fi movies of 2024-2025"
        → "Genres: sci-fi | Release_info: 2024-2025"

        User: "Films starring Tom Hardy and directed by Villeneuve"
        → "Cast: Tom Hardy | Directors: Denis Villeneuve"

        User: "Light-hearted comedy for a relaxing evening"
        → "Plot: light-hearted comedy relaxing"

        User: "Movies with high ratings and strong female leads"
        → "Rating: high | Cast: strong female lead"

        Now, analyze the following user query and rewrite it according to the rules above. Return only the rewritten query.
        """

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

    def generate_with_context(self, query: str, context_chunks: list):
        
        context_text = "\n\n".join(context_chunks)

        system_prompt = (
            "You are an expert movie recommendation assistant with access to a rich movie database.\n"
            "Your task is to provide accurate, relevant and well-explained film recommendations based solely on the given context.\n\n"
            "Rules:\n"
            "- Use ONLY information from the provided context.\n"
            "- Never hallucinate or make up facts.\n"
            "- If the context does not contain enough information, say so honestly.\n"
            "- When recommending films, always include the title and a short reason why it matches the query.\n"
            "- You can recommend 2 to 5 films depending on how well they fit.\n\n"
            "Context:\n"
            f"{context_text}\n\n"
            "Answer in a natural, friendly and engaging tone."
        )

        return self.generate(query, system_prompt=system_prompt)


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