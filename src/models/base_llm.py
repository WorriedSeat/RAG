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

        system_prompt = (
            "You rewrite a user request into ONE concise English retrieval string for a movie FAISS index.\n"
            "The FAISS index embeds documents using two templates:\n"
            "1) plot chunk template (exactly like):\n"
            "Title: {movie_title} Plot: {plot_text}\n"
            "2) meta chunk template (exactly like, built as pieces joined with ' | ')\n"
            "{movie_title} | Release: ... | Rating: ... | Votes: ... | Popularity: ... | Runtime: ... |\n"
            "Genres: ... | Directors: ... | Cast: ... | Production countries: ... | Production companies: ... | Tags: ...\n"
            "You MUST output a single line that concatenates BOTH templates in this exact order, using ' | ' between major parts:\n"
            "Format (must match exactly):\n"
            "Title: {movie_title} Plot: {plot_text} | {movie_title} | Release: {release} | Rating: {rating} | Votes: {votes} | Popularity: {popularity} | Runtime: {runtime} | Genres: {genres} | Directors: {directors} | Cast: {cast} | Production countries: {countries} | Production companies: {companies} | Tags: {tags}\n"
            "Hard rules:\n"
            "- Output ONLY the final one-line string. No JSON, no markdown, no quotes, no extra commentary.\n"
            "- Output must be in English.\n"
            "- Do NOT invent numeric or factual metadata (Release year/Rating/Votes/Popularity/Runtime/Production countries/Production companies) unless the user explicitly provided it.\n"
            "- If a field is not provided by the user, use the exact literal 'Unknown' for that field (e.g., 'Release: Unknown', 'Rating: Unknown', 'Votes: Unknown', etc.).\n"
            "- If the user mentions a specific movie title, set {movie_title} to that title.\n"
            "- {plot_text} must be a short English plot summary / themes / key events matching the user intent (do not add invented cast/director unless user provided them; you can include their names only if they appear in the user request)\n"
            "Output constraints:\n"
            "- The output must be exactly ONE line (no newlines)\n"
        )

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
            "You are a movie recommendation assistant.\n"
            "Use the provided context to answer.\n"
            f"Context:\n{context_text}"
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