"""
Evaluation script: runs queries from data/prep/output.csv through:
  1. Plain LLM (no retrieval)
  2. Full RAG pipeline (classify → search → generate)

Saves results incrementally to data/prep/eval_results.csv
Usage: PYTHONPATH=. .venv/bin/python eval.py
"""

import os
import csv
import traceback
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.dataset.index import FaissIndex
    from src.models.base_llm import BaseLLMModel
    from src.main import RAG
except ImportError:
    from dataset.index import FaissIndex
    from models.base_llm import BaseLLMModel
    from main import RAG

INPUT_PATH  = "data/prep/output.csv"
OUTPUT_PATH = "data/prep/eval_results.csv"

PLAIN_LLM_PROMPT = (
    "You are a helpful movie recommendation assistant. "
    "Recommend movies based on the user's request. "
    "Be concise — 2-4 sentences."
)

FIELDNAMES = [
    "query",
    "search_type",
    "rewritten_query",
    "top5_titles",
    "rag_response",
    "plain_llm_response",
]


def load_queries(path: str) -> list[str]:
    df = pd.read_csv(path, header=None, names=["query"])
    return df["query"].dropna().tolist()


def already_done(path: str) -> set[str]:
    """Return set of queries already saved (for resume support)."""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add(row["query"])
    return done


def run_rag(rag: RAG, query: str) -> dict:
    search_type = rag.classify_query(query)

    if search_type == "meta":
        rewritten = rag.llm.rewrite_query(query)
        search_query = rewritten
    else:
        rewritten = ""
        search_query = query

    candidates = rag.index.search(type=search_type, query=search_query, top_k=20)
    results = rag._filter_results(candidates, top_k=5)
    top5_titles = " | ".join(r["title"] for r in results)
    rag_response = rag.llm.generate_with_context(query, results)

    return {
        "search_type": search_type,
        "rewritten_query": rewritten,
        "top5_titles": top5_titles,
        "rag_response": rag_response,
    }


def run_plain_llm(llm: BaseLLMModel, query: str) -> str:
    return llm.generate(query, system_prompt=PLAIN_LLM_PROMPT, temperature=0.2)


def main():
    queries = load_queries(INPUT_PATH)
    done = already_done(OUTPUT_PATH)
    remaining = [q for q in queries if q not in done]

    print(f"Total queries: {len(queries)} | Already done: {len(done)} | Remaining: {len(remaining)}")

    if not remaining:
        print("All queries already processed.")
        return

    print("Initializing RAG system...")
    rag = RAG()

    write_header = not os.path.exists(OUTPUT_PATH)
    out_file = open(OUTPUT_PATH, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    for i, query in enumerate(remaining, 1):
        print(f"\n[{i}/{len(remaining)}] Query: {query[:80]}...")
        row = {"query": query}
        try:
            rag_data = run_rag(rag, query)
            row.update(rag_data)
            print(f"  search_type={rag_data['search_type']}  top5={rag_data['top5_titles'][:80]}")
        except Exception:
            print(f"  RAG ERROR:\n{traceback.format_exc()}")
            row.update({"search_type": "ERROR", "rewritten_query": "", "top5_titles": "", "rag_response": ""})

        try:
            row["plain_llm_response"] = run_plain_llm(rag.llm, query)
        except Exception:
            print(f"  LLM ERROR:\n{traceback.format_exc()}")
            row["plain_llm_response"] = ""

        writer.writerow(row)
        out_file.flush()
        print(f"  Saved row {i}.")

    out_file.close()
    print(f"\nDone! Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
