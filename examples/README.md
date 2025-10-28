Examples for pdf_agent

This folder contains example utilities and a runnable Jupyter notebook that demonstrate three common workflows:

- Ingesting a folder of PDFs into the local index (ingest)
- Running a retrieval query and saving the JSON result (retrieve)
- Running a comprehensive `analyze_all` job and saving the long-form output (analyze_all)

Files

- `ingest_demo.py` — CLI script that calls `PDFAgent.process_folder` to ingest PDFs.
- `retrieve_demo.py` — CLI script that runs a retrieval query and saves result to `outputs/retrieve_result.json`.
- `analyze_all_demo.py` — CLI script that runs a comprehensive analysis and saves to `outputs/analyze_all_result.txt`.
- `demo.ipynb` — A single notebook that contains the same three demos in separate cells; recommended for interactive exploration.
 - `ingest_demo.ipynb` — Notebook demonstrating ingestion (standalone).
 - `retrieve_demo.ipynb` — Notebook demonstrating retrieval (standalone).
 - `analyze_all_demo.ipynb` — Notebook demonstrating `analyze_all` (standalone).

Quickstart

1. Install dependencies (use your project environment):

```bash
pip install -r requirements.txt
pip install notebook
```

2. Ensure the `outputs/` directory exists (the notebook and scripts write results here):

```bash
mkdir -p outputs
```

3. Run the notebook (interactive):

```bash
jupyter notebook examples/ingest_demo.ipynb
# or
jupyter notebook examples/retrieve_demo.ipynb
# or
jupyter notebook examples/analyze_all_demo.ipynb
```

4. Or run the scripts directly from the command line:

```bash
python3 examples/ingest_demo.py --folder ./downloads/search_results
python3 examples/retrieve_demo.py --query "What are agentic workflows?"
python3 examples/analyze_all_demo.py --query "Summarize recent trends in agentic AI"
```

Notes

- The examples use the public `PDFAgent` API. If the vector index is empty, run the ingest demo first.
- The notebook cells are intentionally simple and include checks to avoid surprising failures.
- For large corpora, embedding generation and the knowledge graph build may take significant time and may require API keys/configuration for Azure/Poe or a running Ollama server.

If you want, I can tidy the notebook for strict linting or split the notebook into smaller tutorial notebooks per blog post.