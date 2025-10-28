"""
examples/analyze_all_demo.py

Run a comprehensive "analyze all" job and save the long-form analysis output to a file.
Usage:
    python3 examples/analyze_all_demo.py --query "Summarize recent trends in agentic AI"
"""
from agents.pdf_agent import PDFAgent
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run analyze_all across the corpus")
    parser.add_argument("--query", type=str, default="Provide a comprehensive literature review of agentic workflows", help="Query to analyze across all documents")
    args = parser.parse_args()

    agent = PDFAgent()

    print("Starting comprehensive analysis (this may take a while)...")
    result = agent.search(args.query, mode="analyze_all")

    answer = result.get("answer", "(no answer)")
    sources = result.get("sources", [])

    # Save to file
    out_path = "outputs/analyze_all_result.txt"
    with open(out_path, "w") as fh:
        fh.write(answer)

    print(f"Analysis saved to {out_path}")
    print(f"Documents analyzed: {len(sources)}")


if __name__ == "__main__":
    main()
