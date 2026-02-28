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

    def generate(self, query: str, system_prompt: str = None):
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": query})

        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response["choices"][0]["message"]["content"]

    def generate_with_context(self, query: str, context_chunks: list):
        """
        RAG-style generation
        """

        context_text = "\n\n".join(context_chunks)

        system_prompt = (
            "You are a movie recommendation assistant.\n"
            "Use the provided context to answer.\n\n"
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