# PDF Research Assistant

An intelligent research assistant that can analyze, search, and chat about your PDF document collection using Azure OpenAI embeddings and Poe LLM (Claude Sonnet 4).

## Features

### Core Capabilities
- 📄 **PDF Processing**: Automatically index and process PDF documents
- 🔍 **Intelligent Search**: Vector-based semantic search across documents
- 💬 **Interactive Chat**: Conversational interface with context awareness
- 🧠 **Memory**: Maintains conversation history and context
- 🔄 **Auto-Watch**: Monitors directory for new PDFs and auto-indexes them

### Search Modes

#### 1. Enhanced Search (Default)
- Uses similarity-based retrieval
- Returns top 30 most relevant chunks
- Best for specific questions and targeted research
- Response time: ~30 seconds

#### 2. **Comprehensive Analysis (NEW!)** ⭐
- Analyzes **ALL documents** in your knowledge base
- Processes in batches to avoid token limits
- Ensures complete coverage of your collection
- Response time: ~4 minutes for 44 papers
- Automatically activated by keywords like "analyze all", "every paper", etc.

## Installation

### Prerequisites
- Python 3.13+
- Azure OpenAI account (for embeddings)
- Poe API key (for LLM completions)

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd pdf_agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure settings:
Create/edit `config/settings.py`:
```python
# Azure OpenAI (for embeddings)
AZURE_OPENAI_ENDPOINT = "https://YOUR-RESOURCE.openai.azure.com"
AZURE_OPENAI_API_KEY = "your-key-here"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding-3-large"
AZURE_OPENAI_EMBEDDING_API_VERSION = "2024-08-01-preview"

# Poe (for LLM)
POE_API_BASE_URL = "https://api.poe.com/v1"
POE_API_KEY = "your-poe-key-here"
POE_MODEL = "Claude-Sonnet-4"

# Storage paths
PDF_WATCH_DIR = Path.home() / "Downloads" / "search_results"
VECTOR_STORE_PATH = Path(__file__).parent.parent / "storage" / "vector_store"
```

4. Create storage directories:
```bash
mkdir -p storage/{vector_store,chat_history,graph_store}
```

## Usage

### Quick Start

```python
from agents.pdf_agent import PDFAgent

# Initialize agent
agent = PDFAgent()

# Add PDFs to your watch directory
# They'll be automatically indexed

# Ask questions
response = agent.chat("What are the main themes in these papers?")
print(response)

# Analyze ALL papers comprehensively
response = agent.chat("analyze all papers")
print(response)
```

### Command Line

```bash
# Start interactive mode
python main.py

# Example session:
> analyze all papers on agent architectures
> what papers discuss reinforcement learning?
> give me a comprehensive summary of every document
```

### Comprehensive Analysis

To analyze **all documents** (not just top similar ones):

```python
# Any of these trigger comprehensive mode:
agent.chat("analyze all papers")
agent.chat("provide a comprehensive review of all documents")
agent.chat("summarize every paper in the collection")
agent.chat("analyze each paper individually")
```

**What happens:**
1. System detects "analyze all" keywords
2. Retrieves chunks from ALL unique documents (not just top-k)
3. Processes in batches of 10 papers
4. Returns comprehensive analysis of entire collection

**See:** [ANALYZE_ALL_QUICK_START.md](ANALYZE_ALL_QUICK_START.md) for details

### Advanced Usage

```python
from agents.pdf_agent import PDFAgent
from core.search_engine import SearchEngine

agent = PDFAgent()

# Different search modes
agent.search("query", mode="simple")        # Basic vector search
agent.search("query", mode="enhanced")      # With conversation context
agent.search("query", mode="summarize")     # Get summary of results
agent.search("query", mode="analyze_all")   # Analyze ALL documents

# Get sources
sources = agent.get_sources("quantum computing")

# View conversation history
history = agent.get_conversation_history(limit=10)

# Clear memory
agent.clear_memory()
```

## Architecture

```
pdf_agent/
├── agents/
│   └── pdf_agent.py          # Main agent interface
├── config/
│   └── settings.py           # Configuration with provider selection
├── core/
│   ├── azure_openai_wrapper.py  # Azure OpenAI embeddings + Poe LLM
│   ├── ollama_wrapper.py        # Ollama embeddings + LLM (NEW)
│   ├── context_manager.py       # Document retrieval
│   ├── memory_manager.py        # Conversation memory
│   ├── pdf_parser.py            # PDF processing with provider switching
│   └── search_engine.py         # Search logic
├── storage/
│   ├── chat_history/         # Conversation storage
│   ├── graph_store/          # Graph-based indexes
│   └── vector_store/         # ChromaDB vector embeddings
├── utils/
│   ├── file_watcher.py       # Auto-indexing new PDFs
│   └── logger.py             # Logging utilities
└── test_ollama.py            # Ollama integration test (NEW)
```

### AI Provider Selection

The system supports multiple AI providers for embeddings and chat:

#### Embedding Providers
- **Azure OpenAI** (default): `text-embedding-3-large` (3072 dimensions)
- **Ollama**: Local `nomic-embed-text` model

#### LLM Providers  
- **Poe** (default): Claude Sonnet 4 via Poe API
- **Ollama**: Local models like `llama2`, `mistral`, etc.

#### Switching Providers

```bash
# Check current providers
providers

# Switch embedding provider
set embedding ollama
set embedding azure

# Switch LLM provider  
set llm ollama
set llm poe
```

**Note:** Provider changes require application restart to take effect.

#### Ollama Setup

1. Install Ollama: https://ollama.ai/download
2. Start Ollama server
3. Pull required models:
```bash
ollama pull nomic-embed-text  # For embeddings
ollama pull llama2           # For chat (or your preferred model)
```
4. Test installation: `python test_ollama.py`

## Performance

### Current Knowledge Base
- **Documents indexed**: 44 PDFs
- **Total chunks**: 1,400
- **Embedding model**: text-embedding-3-large (3072-dim)
- **LLM**: Claude Sonnet 4 via Poe

### Benchmarks
- **Enhanced search**: ~30 seconds, 30 chunks, 10-15 papers
- **Comprehensive analysis**: ~4 minutes, 88 chunks, all 44 papers
- **Indexing new PDF**: ~5-10 seconds per document

## Recent Updates

### ✅ Comprehensive Analysis Feature (2025-10-22)
**Problem:** "Analyze all papers" only returned 5 papers instead of all 44

**Solution:**
- Added `retrieve_all_documents()` to get chunks from ALL papers
- Implemented batched processing (10 papers/batch)
- Auto-detection of comprehensive analysis requests
- Full coverage of knowledge base

**Files Modified:**
- `core/context_manager.py` - Added retrieve_all_documents()
- `core/search_engine.py` - Added analyze_all mode with batching
- `agents/pdf_agent.py` - Added keyword detection

**Documentation:**
- [COMPREHENSIVE_ANALYSIS_FIX.md](COMPREHENSIVE_ANALYSIS_FIX.md) - Technical details
- [ANALYZE_ALL_QUICK_START.md](ANALYZE_ALL_QUICK_START.md) - User guide

## Troubleshooting

### Missing Papers
If some PDFs aren't being analyzed:
```python
# Check which PDFs are indexed
python -c "
from agents.pdf_agent import PDFAgent
agent = PDFAgent()
agent.list_indexed_documents()
"
```

### Slow Responses
- Comprehensive analysis takes 3-5 minutes (normal)
- Enhanced search should be <30 seconds
- Check agent.log for errors

### Connection Timeouts
- Some batches may timeout (check logs)
- Processing continues automatically
- Final response includes all successful batches

## Logs

All operations logged to `agent.log`:
```bash
tail -f agent.log
```

## Documentation

- [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - Migration from ServiceContext to Settings
- [COMPREHENSIVE_ANALYSIS_FIX.md](COMPREHENSIVE_ANALYSIS_FIX.md) - How analyze_all works
- [ANALYZE_ALL_QUICK_START.md](ANALYZE_ALL_QUICK_START.md) - Quick start guide

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## License

[Your License Here]

## Support

For issues and questions:
- Check agent.log for errors
- Review documentation in /docs
- Open GitHub issue with logs

---

**Status**: ✅ Production Ready  
**Last Updated**: 2025-10-22  
**Version**: 2.0 (with comprehensive analysis)
