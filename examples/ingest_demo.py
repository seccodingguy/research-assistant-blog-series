"""
examples/ingest_demo.py

Simple script to ingest PDFs from a folder using the PDFAgent API.
Usage:
    python3 examples/ingest_demo.py --folder ./downloads/search_results

This will parse and index any PDFs it finds and print processing statistics.
"""
from pathlib import Path
import argparse
import json

from agents.pdf_agent import PDFAgent


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into pdf_agent index")
    parser.add_argument(
        "--folder",
        type=str,
        default="downloads/search_results",
        help="Folder containing PDFs to ingest"
    )

    args = parser.parse_args()
    folder = Path(args.folder)

    if not folder.exists():
        print(f"Folder not found: {folder.resolve()}")
        return

    agent = PDFAgent()
    print("Starting folder ingest... this may take a few minutes depending on corpus size")

    result = agent.process_folder(folder)

    print("--- Ingest Result ---")
    print(json.dumps(result, indent=2))

    stats = agent.get_stats()
    print("--- Agent Stats ---")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
