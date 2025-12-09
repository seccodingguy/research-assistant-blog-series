# PDF Agent User Guide

**Version:** 2.3.1  
**Last Updated:** October 30, 2025

---

## Table of Contents

1. [Knowledge Graph Integration](#1-knowledge-graph-integration)
2. [Paper Search Integration](#2-paper-search-integration)
3. [Response Save Functionality](#3-response-save-functionality)
4. [Comprehensive Analysis (Analyze All)](#4-comprehensive-analysis-analyze-all)
5. [API Blocking Reference](#5-api-blocking-reference)

---

## 1. Knowledge Graph Integration

### Overview

The PDF Agent includes a **Knowledge Graph** feature that enhances context retrieval and relationship mapping between research documents. The graph store works alongside the existing vector store to provide hybrid retrieval with deeper insights into document relationships.

### Features

#### 1. Hybrid Vector + Graph Retrieval
- Combines traditional vector similarity search with knowledge graph relationships
- Automatically enriches search results with graph-based context
- Provides better understanding of connections between concepts

#### 2. LLM-Based Entity Extraction
- Uses your configured LLM to extract entities and relationships from documents
- Supports scientific concept types (AI, ML, agents, embeddings, etc.)
- Relationship types: causes, correlates_with, inhibits, enhances, supports, interacts_with, associated_with

#### 3. Graph Visualization
- Generate JSON visualization data for knowledge graphs
- View top connected concepts and their relationships
- Filter graphs by query to see relevant subgraphs

#### 4. Graph Queries
- Query the knowledge graph directly for relationship-based insights
- Get related concepts and their connections
- Traverse the graph to discover indirect relationships

### Architecture

```
┌─────────────────┐
│  PDF Documents  │
└────────┬────────┘
         │
         ↓
┌─────────────────┐      ┌──────────────────┐
│   PDF Parser    │─────→│  GraphManager    │
│  (Chunking)     │      │ (LLM Extraction) │
└────────┬────────┘      └────────┬─────────┘
         │                        │
         ↓                        ↓
┌─────────────────┐      ┌──────────────────┐
│  Vector Store   │      │   Graph Store    │
│   (ChromaDB)    │      │ (SimpleGraph +   │
│                 │      │   NetworkX)      │
└────────┬────────┘      └────────┬─────────┘
         │                        │
         └────────┬───────────────┘
                  ↓
         ┌─────────────────┐
         │ Context Manager │
         │ (Hybrid Search) │
         └─────────────────┘
```

### Building the Graph from Existing Documents

Run the build script to process your existing documents:

```bash
python3 build_knowledge_graph.py
```

This will:
1. Connect to your existing ChromaDB vector store
2. Extract all documents from the collection
3. Use your configured LLM to extract entities and relationships
4. Build and persist the knowledge graph
5. Generate initial visualization data

**Note**: This process may take 30-60 minutes for 1,400 documents as it requires LLM calls for each document batch.

### CLI Commands

#### View Graph Statistics
```bash
graph stats
```

Shows:
- Total nodes and edges
- Average degree (connections per node)
- Graph density
- Node type distribution

#### Visualize Graph
```bash
graph viz
```

Generates `./outputs/graph_viz.json` with visualization data including:
- Top connected concepts
- Node and edge information
- Relationship types

#### Query the Graph
```bash
graph query <your query>
```

Example:
```bash
graph query agentic workflows
```

Returns:
- Graph-based answer
- Related nodes (concepts)
- Relationships found

### Programmatic Usage

```python
from core.graph_manager import GraphManager

# Initialize
graph_mgr = GraphManager()

# Build from documents
graph_mgr.build_graph_from_documents(
    documents,
    max_triplets_per_chunk=10
)

# Query
result = graph_mgr.query_graph("agentic AI")
print(result['response'])
print(f"Found {len(result['nodes'])} related concepts")

# Get statistics
stats = graph_mgr.get_graph_statistics()
print(f"Nodes: {stats['total_nodes']}, Edges: {stats['total_edges']}")

# Visualize
viz_data = graph_mgr.visualize_graph(
    output_path="./outputs/graph.json",
    max_nodes=50
)
```

### Concept Types

The system recognizes these scientific concept types:

- **AI & ML**: artificial intelligence, machine learning, deep learning
- **Models**: neural networks, transformers, LLMs
- **Techniques**: supervised, unsupervised, reinforcement learning
- **Components**: embeddings, prompts, context, memory
- **Applications**: computer vision, NLP, robotics
- **Agents**: agent, agentic, reasoning
- **Metrics**: accuracy, performance
- **Data**: datasets, knowledge graphs
- **Issues**: hallucination, bias

### Relationship Types

Extracted relationships include:

- **causes**: X causes Y
- **correlates_with**: X correlates with Y
- **inhibits**: X inhibits Y
- **enhances**: X enhances Y
- **supports**: X supports Y
- **interacts_with**: X interacts with Y
- **associated_with**: X is associated with Y

### Storage

#### Graph Store Location
```
./storage/graph_store/
  ├── graph_store.json          # Main graph data
  └── docstore.json             # Document metadata
```

#### Visualization Outputs
```
./outputs/
  ├── graph_viz.json            # Full graph visualization
  └── initial_graph_viz.json    # Post-build visualization
```

### Performance Considerations

**Build Time:**
- Per Document: ~2-5 seconds (LLM extraction)
- 1,400 Documents: ~45-90 minutes total
- Recommendation: Run once, then incremental updates

**Query Time:**
- Vector Search: ~100-500ms
- Graph Query: ~50-200ms
- Hybrid Search: ~200-700ms (parallel retrieval)

**Memory Usage:**
- Graph Store: ~10-50MB for 1,400 documents
- NetworkX Graph: ~20-100MB in memory
- ChromaDB: Unchanged (~500MB-2GB)

### Troubleshooting

**Issue: Graph Not Building**
- Check LLM configuration (Poe or Ollama must be set)
- Verify API keys in `system_config.json`
- Check logs for specific errors

**Issue: Empty Graph**
- Check LLM is responding (test with simple query)
- Verify documents have text content
- Increase `max_triplets_per_chunk`
- Check logs for extraction errors

**Issue: Slow Queries**
- Reduce `max_nodes` in visualization (default: 100)
- Use more specific queries
- Filter subgraphs by concept type

---

## 2. Paper Search Integration

### Overview

The Paper Search Service allows you to:
- Search for academic papers across multiple databases (arXiv, Semantic Scholar, CrossRef, etc.)
- Download PDFs automatically
- Organize downloaded papers with metadata
- Track search statistics

### Quick Start

#### Using the CLI

Start the main application:
```bash
python3 main.py
```

Then use the `papers` command:

**Search for Papers:**
```
papers search machine learning transformers
```

**Search and Download Papers:**
```
papers download neural networks
```

**View Available Search Engines:**
```
papers sources
```

**View Search Statistics:**
```
papers stats
```

### Available Search Engines

| Engine | Default | Coverage | PDF Availability |
|--------|---------|----------|------------------|
| arXiv | ✓ | Physics, Math, CS | High |
| Semantic Scholar | ✓ | Multidisciplinary | Medium |
| CrossRef | ✓ | DOI Registry | Low |
| DuckDuckGo | ✗ | General | Medium |
| Google Scholar | ✗ | General | Medium |

### Programmatic Usage

#### Basic Search Example

```python
import asyncio
from services.paper_search_service import PaperSearchService

async def search_papers():
    # Initialize service
    service = PaperSearchService(download_directory="./downloads/papers")
    await service.initialize()
    
    try:
        # Search for papers
        results = await service.search(
            query="deep learning computer vision",
            max_results_per_source=10,
            download_pdfs=False
        )
        
        print(f"Found {results['total_results']} papers")
        
        for paper in results['results']:
            print(f"Title: {paper['title']}")
            print(f"Authors: {', '.join(paper['authors'])}")
            print(f"Year: {paper['year']}")
            print(f"DOI: {paper['doi']}")
            print(f"PDF: {paper['pdf_url']}")
            print()
    
    finally:
        await service.shutdown()

asyncio.run(search_papers())
```

#### Search and Download Example

```python
async def download_papers():
    service = PaperSearchService()
    await service.initialize()
    
    try:
        results = await service.search_and_download(
            query="quantum computing algorithms",
            sources=['arxiv', 'semantic_scholar'],
            max_results=5,
            max_concurrent_downloads=3
        )
        
        print(f"Papers found: {results['total_results']}")
        
        if 'downloads' in results:
            downloads = results['downloads']
            print(f"Downloaded: {downloads['successful']}")
            print(f"Failed: {downloads['failed']}")
    
    finally:
        await service.shutdown()

asyncio.run(download_papers())
```

#### Search by DOI

```python
async def download_by_doi():
    service = PaperSearchService()
    await service.initialize()
    
    try:
        result = await service.download_by_doi(
            doi="10.1038/nature14539",
            subfolder="nature_papers"
        )
        
        if result['success']:
            print(f"Downloaded: {result['paper']['title']}")
            print(f"Path: {result['download']['path']}")
    
    finally:
        await service.shutdown()

asyncio.run(download_by_doi())
```

### Configuration

Edit `system_config.json`:

```json
{
  "search": {
    "max_results_per_source": 10,
    "timeout_seconds": 30,
    "enabled_sources": [
      "arxiv",
      "semantic_scholar",
      "crossref"
    ],
    "semantic_scholar_api_key": "your_api_key_here"
  }
}
```

Or configure programmatically:

```python
service = PaperSearchService()

# Enable/disable sources
service.enable_source('google_scholar')
service.disable_source('crossref')

# Check enabled sources
enabled = service.get_enabled_sources()
```

### Download Organization

```
downloads/
├── search_results/
│   ├── 20241029_143022_machine_learning/
│   │   ├── Paper_Title_abc123.pdf
│   │   ├── Another_Paper_def456.pdf
│   │   └── ...
│   ├── metadata/
│   │   ├── search_20241029_143022_machine_learning.json
│   │   ├── Paper_Title_abc123.json
│   │   └── Another_Paper_def456.json
│   ├── by_date/
│   └── by_source/
```

### Integration with PDF Agent

Complete workflow:
```bash
# 1. Find papers
papers download transformers attention mechanism

# 2. Process downloads
process ./downloads/search_results/20241029_143022_transformers_attention

# 3. Query knowledge base
search what are the key components of transformer architecture?
```

### Examples

Run the examples:

```bash
# Simple example
python3 examples/simple_paper_search.py

# Full interactive demo
python3 examples/search_papers_demo.py
```

### Troubleshooting

**No results found:**
- Try broader search terms
- Check enabled sources: `papers sources`
- Verify network connectivity

**Downloads failing:**
- Check PDF URL availability
- Verify file size limits (default 100MB)
- Check download directory permissions

**Rate limiting errors:**
- Reduce max_results_per_source
- Add delays between searches
- Consider API keys for higher limits

---

## 3. Response Save Functionality

### Overview

The PDF Agent includes comprehensive response saving functionality for both `search` and `chat` commands. Responses can be saved in multiple formats with full metadata preservation.

### Supported Formats

- **TXT** - Plain text format with structured sections
- **MD** - Markdown format with formatting
- **JSON** - Machine-readable format with all metadata
- **HTML** - Styled web page with professional formatting
- **CSV** - Spreadsheet format for data analysis

### What Gets Saved

For each response:
- **Original Query** - Your search/chat query
- **LLM Response** - The complete answer
- **Sources** - List of source documents with relevance scores
- **Metadata** - Timestamp, mode, chunks retrieved, documents analyzed
- **Batch Information** - For comprehensive analyses (analyze_all mode)

### Basic Workflow

1. **Execute a search or chat command:**
   ```
   search What are AI agent architectures?
   ```

2. **View the response** in the terminal

3. **Save prompt appears:**
   ```
   Save this response? [y/n] (n):
   ```

4. **Choose to save** by typing `y`

5. **Enter file path:**
   ```
   File path: ./my_responses/agent_architectures.md
   ```

6. **Optional: Add timestamp:**
   ```
   Add timestamp to filename? [y/n] (n): y
   ```
   This saves as: `agent_architectures_20251024_232440.md`

### Path Examples

**Relative paths:**
```
./outputs/response.txt
../results/analysis.json
responses/research.html
```

**Absolute paths:**
```
/home/username/documents/ai_research.md
~/Downloads/agent_response.csv
```

**Windows paths (in WSL):**
```
/mnt/c/Users/username/Documents/response.txt
```

### Format Selection

The format is automatically determined from the file extension:
- `.txt` → Plain text
- `.md` → Markdown
- `.json` → JSON
- `.html` → HTML
- `.csv` → CSV

### Quick Reference

| Format | Extension | Best For |
|--------|-----------|----------|
| Plain Text | `.txt` | Quick reference, grep-able |
| Markdown | `.md` | Documentation, GitHub |
| JSON | `.json` | APIs, data processing |
| HTML | `.html` | Sharing, presentations |
| CSV | `.csv` | Spreadsheets, analysis |

### Examples

**Example 1: Quick Save**
```
You: search What is reinforcement learning?
[Response displays]

Save this response? [y/n] (n): y
File path: ./rl_overview.txt
Add timestamp to filename? [y/n] (n): n

✓ Response saved successfully to:
  /home/user/pdf_agent/rl_overview.txt
```

**Example 2: JSON for Processing**
```
You: search Deep learning applications
[Response displays]

Save this response? [y/n] (n): y
File path: ./data/dl_apps.json
Add timestamp to filename? [y/n] (n): n

✓ Response saved successfully to:
  /home/user/pdf_agent/data/dl_apps.json
```

### Tips

1. **Create organized directories:**
   ```bash
   mkdir -p ./outputs/{txt,md,json,html,csv}
   ```

2. **Use descriptive filenames:**
   ```
   ./outputs/md/agent_architectures_overview.md
   ```

3. **Enable auto-timestamping for version control:**
   ```
   Add timestamp to filename? [y/n] (n): y
   ```

4. **Use JSON format for programmatic processing:**
   ```python
   import json
   with open('response.json', 'r') as f:
       data = json.load(f)
       print(data['answer'])
   ```

5. **Decline to save by pressing Enter:**
   ```
   Save this response? [y/n] (n): [Enter]
   ```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Unsupported format | Use: txt, md, json, html, csv |
| Permission denied | Check directory permissions |
| Directory not found | Will be created automatically |
| Want to cancel | Press Ctrl+C or leave path blank |

---

## 4. Comprehensive Analysis (Analyze All)

### Overview

The PDF Research Assistant supports comprehensive analysis of **ALL documents** in your knowledge base, not just the most similar ones.

### How to Use

#### Method 1: Natural Language (Recommended)

Simply use phrases like:
```python
from agents.pdf_agent import PDFAgent

agent = PDFAgent()

# These all trigger comprehensive analysis:
agent.chat("analyze all papers")
agent.chat("give me a summary of all documents")
agent.chat("provide a comprehensive review of every paper")
agent.chat("analyze each paper in the collection")
```

#### Method 2: Direct Mode

```python
from core.search_engine import SearchEngine

engine = SearchEngine()
result = engine.search("your question", mode="analyze_all")
print(result["answer"])
```

### What Happens

1. **Auto-Detection:** System detects "analyze all" keywords
2. **Full Retrieval:** Gets chunks from ALL indexed papers
3. **Batched Processing:** Processes in groups of 10 papers
4. **Progress Logging:** Shows batch progress in logs
5. **Comprehensive Response:** Returns analysis of all papers (~40KB text)

### Example Output

```
COMPREHENSIVE ANALYSIS OF ALL 44 RESEARCH PAPERS

Total papers analyzed: 44
Processed in 5 batches

=== BATCH 1: Papers 1-10 ===

## Paper 1: Agent²: An Agent-Generates-Agent Framework...
Main research question: ...
Key methodology: ...
Main findings: ...

## Paper 2: Agentic Visualization...
[...]

=== BATCH 2: Papers 11-20 ===
[...]

Summary: Successfully analyzed all 44 papers in the knowledge base.
```

### Performance

- **Processing Time:** ~4 minutes for 44 papers
- **Response Size:** ~40,000 characters
- **Coverage:** 100% of indexed documents
- **Batches:** 5 (10 papers each, except last)

### When to Use

**Use "Analyze All" Mode When:**
- ✅ You want a comprehensive literature review
- ✅ You need to see ALL papers on a topic
- ✅ You're doing corpus-wide analysis
- ✅ You want to ensure nothing is missed
- ✅ You need complete coverage

**Use Normal Search When:**
- ✅ You have a specific question
- ✅ You want quick answers (~30 seconds)
- ✅ You're looking for most relevant papers only
- ✅ You don't need exhaustive coverage

### Command Line

```bash
cd /mnt/c/Users/mwireman/repos/pdf_agent

# Interactive mode
python main.py
> analyze all papers

# Or programmatically
python -c "
from agents.pdf_agent import PDFAgent
agent = PDFAgent()
response = agent.chat('analyze all papers')
print(response)
"
```

### Logs

Watch progress in real-time:
```bash
tail -f agent.log
```

You'll see:
```
INFO | Detected request for comprehensive analysis of all documents
INFO | Found 44 unique documents
INFO | Retrieved 88 chunks from 44 documents
INFO | Processing batch 1: papers 1-10
INFO | Processing batch 2: papers 11-20
...
INFO | Analysis complete for 44 documents
```

### Trigger Keywords

Any of these in your query will activate analyze_all mode:
- "analyze all"
- "all papers"
- "all documents"
- "every paper"
- "comprehensive review"
- "analyze each"
- "all the papers"
- "each paper"
- "all research"

### Troubleshooting

**"Only 5 papers in response"**
- Make sure your query includes keywords like "all", "every", or "comprehensive"
- Or use `mode="analyze_all"` explicitly

**"Connection timeout"**
- Normal for large batches (batch 2-3 may timeout but continue)
- System continues processing remaining batches
- Check logs for completion

---

## 5. API Blocking Reference

### Summary

All LLM and Embedding API calls now **properly block** and wait for complete responses before proceeding. No async operations, no premature returns.

### Key Features

**✓ Synchronous Blocking**
- All API calls use blocking `requests.post()`
- Code execution pauses until response received
- No race conditions or incomplete data

**✓ Extended Timeouts**
- **LLM calls**: Up to 10 minutes (600 seconds)
- **Embedding calls**: Up to 2 minutes (120 seconds)
- Separate connect (10-15s) and read timeouts

**✓ Retry Logic**
- 3 automatic retry attempts
- Exponential backoff: 2s → 4s → 8s
- Handles timeout, connection, and request errors

**✓ Response Validation**
- Checks for empty responses
- Validates data structure
- Retries on invalid responses

**✓ Enhanced Logging**
```
INFO - LLM API call (attempt 1/3) - blocking until response
INFO - LLM response received in 67.80s (190 chars)
```

### Usage Examples

**Embedding Call (Blocks ~7-10s):**
```python
from core.ollama_wrapper import OllamaEmbedding

embedding = OllamaEmbedding(base_url="http://server:11434")
result = embedding._get_query_embedding("sample text")
# ^^^ BLOCKS HERE until embedding received
print(f"Got embedding: {len(result)} dimensions")
```

**LLM Call (Blocks ~1-60s):**
```python
from core.ollama_wrapper import OllamaLLM

llm = OllamaLLM(base_url="http://server:11434")
response = llm.complete("What is AI?")
# ^^^ BLOCKS HERE until response received
print(response.text)
```

**Batch Processing (Blocks for all items):**
```python
texts = ["text1", "text2", "text3"]
embeddings = embedding._get_text_embeddings(texts)
# ^^^ BLOCKS until ALL embeddings received
print(f"Processed {len(embeddings)} items")
```

### Performance Characteristics

**Typical Response Times:**
- **First LLM call**: 10-15s (model loading)
- **Subsequent LLM calls**: 1-5s (model cached)
- **Single embedding**: 7-10s
- **Batch embeddings**: ~1.2s per item

**Batch Processing:**
- Batch size: 5 items (Ollama), 100 items (Azure)
- Inter-batch delay: 0.5s (Ollama only)
- Progress logged for each batch

### Common Issues

**Q: Call is taking too long**
- Normal for first call (model loading)
- Check server load and model size
- Consider smaller model or reduced context

**Q: Getting timeout errors**
- Increase timeout in code if needed
- Check network connectivity
- Verify server is responding

**Q: Empty responses**
- Automatic retry will attempt 3 times
- Check model is loaded correctly
- Verify prompt is valid

---

## Conclusion

This user guide covers the major features of the PDF Agent system. For more detailed technical information, see:

- **DEVELOPMENT_HISTORY.md** - Complete development changelog
- **ARCHITECTURE.md** - System architecture details
- **TECHNICAL_SPECIFICATION.md** - Technical specifications

For support:
1. Check logs in `./logs/agent.log`
2. Review configuration in `system_config.json`
3. Run example scripts in `examples/`
4. Check test results with pytest

**Version:** 2.3.1  
**Status:** Production Ready  
**Last Updated:** October 30, 2025
