"""
examples/retrieve_demo.py

Simple script to run a query against the indexed corpus and print the answer + sources.
Usage:
    python3 examples/retrieve_demo.py --query "What are agentic workflows?"
"""
from agents.pdf_agent import PDFAgent
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Run a retrieval query against pdf_agent")
    parser.add_argument("--query", type=str, default="What are agentic workflows?", help="Query text")
    parser.add_argument("--mode", type=str, default="enhanced", help="Search mode: simple|enhanced|summarize|analyze_all")
    args = parser.parse_args()

    agent = PDFAgent()

    try:
        result = agent.search(args.query, mode=args.mode)
    except RuntimeError as e:
        print("Search not ready: index may be empty. Run the ingestion demo first.")
        raise

    print("--- Answer ---")
    print(result.get("answer", "(no answer)"))

    print("\n--- Sources ---")
    for src in result.get("sources", []):
        # Each source is typically a dict with file_name and file_path
        if isinstance(src, dict):
            fname = src.get("file_name") or src.get("id") or str(src)
            fpath = src.get("file_path", "")
            print(f"- {fname} ({fpath})")
        else:
            print(f"- {src}")

    # Optionally save to JSON file
    with open("outputs/retrieve_result.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print("\nResult saved to outputs/retrieve_result.json")


if __name__ == "__main__":
    main()
