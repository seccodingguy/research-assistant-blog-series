# Functional Design Document (FDD)
# PDF Agent - Intelligent Document Assistant

## Document Information

- **Version**: 1.3
- **Date**: October 31, 2025
- **Author**: Senior Software Architect
- **System**: PDF Research Assistant Agent
- **Document Type**: Functional Design Document

---

## Project Scope

### Project Objectives

The PDF Agent project aims to create an intelligent research assistant that leverages advanced AI technologies to provide comprehensive analysis and search capabilities across PDF document collections. The system will enable researchers, analysts, and knowledge workers to efficiently process, search, and analyze large volumes of PDF documents using natural language queries.

### Project Boundaries

#### In Scope
- PDF document processing and indexing
- Vector-based semantic search capabilities
- Conversational AI interface with context awareness
- Multi-provider AI support (Azure OpenAI + Ollama)
- Memory management for conversation history
- File system monitoring for automatic document ingestion
- Command-line interface for user interaction
- Comprehensive analysis across entire document collections
- Provider selection and switching capabilities
- Knowledge graph construction and entity classification
- Multi-phase classification strategy (pattern/keyword/LLM-based)
- Graph-based entity resolution and relationship normalization

#### Out of Scope
- Web-based user interface (future enhancement)
- Multi-user support and authentication
- Integration with external document management systems
- Support for non-PDF document formats
- Real-time collaboration features
- Mobile application development
- Advanced data visualization and analytics dashboards

### Key Features and Functionalities

#### Core Features
1. **Document Processing**: Automatic PDF parsing, text extraction, and vector indexing
2. **Intelligent Search**: Multi-mode search with semantic understanding
3. **Conversational Interface**: Context-aware chat with memory retention
4. **Provider Flexibility**: Support for both cloud (Azure/Poe) and local (Ollama) AI providers
5. **Auto-Indexing**: Real-time monitoring and processing of new documents
6. **Comprehensive Analysis**: Batch processing across entire document collections
7. **Knowledge Graph Construction**: Entity extraction and relationship mapping with multi-phase classification
8. **Graph-Based Classification**: YAML-configured classification with pattern matching, keyword lookup, and LLM-based semantic classification

## Recent updates (2025-10-31)

This Functional Design Document has been updated to reflect recent code and architectural changes across the project. Key updates summarized below — see the implementation in `core/` and the documentation directory for details.

- **Knowledge Graph Classification Enhancement (2025-10-31)**: Implemented 4-phase classification strategy achieving 45.4% classification accuracy (up from 24.3% baseline). Phase 1+2 introduced YAML-based external configuration (`config/graph_ontology.yaml`) with 48 ConceptTypes, 254+ keywords, and 9 base patterns. Phase 3 added 18 domain-specific patterns for AI/ML, security, and distributed systems. Phase 4 implemented hybrid LLM classification with non-concept filtering (15 patterns), context-aware LLM classification, and persistent caching. System now supports graph operations via CLI: `graph reclassify-hybrid`, `graph merge`, `graph normalize`, `ontology stats`. See `documentation/PHASE_3_SUMMARY.md` for details.
- **Async Event Loop Fix (2025-10-30)**: Integrated `nest_asyncio` to enable nested event loops, allowing synchronous LLM calls (knowledge graph building) within async workflow contexts. This resolves "Cannot run the event loop while another loop is running" errors during the process action in async workflows. The fix enables seamless integration of async workflows (paper search, download) with synchronous operations (PDF processing, graph building). See `documentation/EVENT_LOOP_FIX.md` and updated architecture documentation for complete technical details.
- **Hybrid Async/Sync Architecture**: System now supports async workflows (`execute_paper_workflow`) that integrate synchronously with knowledge graph building and PDF processing without event loop conflicts.
- **LlamaIndex API migration**: Migrated from the older ServiceContext pattern to the newer Settings-based configuration and updated associated components for compatibility with LlamaIndex 0.14+ (see `core/pdf_parser.py`, `core/search_engine.py`).
- **Azure OpenAI & Poe integration**: Added `core/azure_openai_wrapper.py` to centralize Azure embeddings and add Poe LLM support for cloud completions; fixed endpoint handling, batching, and error recovery.
- **Ollama local provider**: Stabilized local Ollama integration for both embeddings and LLM completions (`core/ollama_wrapper.py`) to support privacy-focused, offline workflows.
- **Memory and ChatMessage fixes**: Resolved ChatMessage attribute handling and ensured memory manager stores/retrieves messages safely (`core/memory_manager.py`, `core/search_engine.py`).
- **Retrieval improvements**: Increased default retrieval `top_k`, lowered similarity cutoff for broader coverage, removed duplicate retrieval calls, and enhanced prompt templates for comprehensive analysis (`config/settings.py`, `core/context_manager.py`).
- **Knowledge Graph integration**: Added `core/graph_manager.py` and `build_knowledge_graph.py` to extract entities/relations via the configured LLM, persist a graph store, and enable hybrid vector+graph retrieval with multi-phase classification (see `documentation/KNOWLEDGE_GRAPH_GUIDE.md`).
- **Dependency updates and environment fixes**: Updated `requirements.txt` (including `nest-asyncio>=1.6.0`, NetworkX, PyYAML) and settings loader for Python 3.13 compatibility and improved installation reliability.

Quick verification steps:
1. Confirm provider configuration in `system_config.json` / environment variables (Azure, Poe, or Ollama).
2. Run `python3 main.py` and run a sample search to confirm multi-source citations in results.
3. Test async workflows: Use natural language like "search, download, and process papers about X" to verify process action works without event loop errors.
4. Optionally rebuild the knowledge graph with `python3 build_knowledge_graph.py` (may take significant time for large corpora).
5. Test graph classification: Run `graph reclassify-hybrid --dry-run` to preview hybrid LLM classification or `ontology stats` to view current rules.


#### Secondary Features
1. **Memory Management**: Persistent conversation history and session management
2. **Source Attribution**: Document source tracking and citation
3. **Configuration Management**: Runtime provider switching and settings management
4. **Error Handling**: Comprehensive error reporting and recovery mechanisms
5. **Performance Monitoring**: System statistics and health monitoring
6. **Entity Resolution**: Duplicate entity merging with configurable similarity thresholds
7. **Relationship Normalization**: Natural language relationship mapping to standard types
8. **Graph Visualization**: Interactive graph exploration and visualization
9. **Classification Caching**: Persistent LLM classification result caching for cost optimization

### Success Criteria

- Process 100+ PDF documents with <5% error rate
- Provide search responses within 30 seconds for standard queries
- Support comprehensive analysis of 50+ documents within 5 minutes
- Maintain 99% system availability during normal operations
- Enable seamless switching between AI providers
- Provide clear error messages and recovery procedures

---

## Risks and Assumptions

### Technical Risks

| Risk | Impact | Probability | Mitigation Strategy |
|------|--------|-------------|-------------------|
| AI Provider API Changes | High | Medium | Implement abstraction layers and version management |
| Large Document Processing Memory Issues | Medium | Low | Implement chunking and streaming processing |
| Vector Store Performance Degradation | High | Low | Regular maintenance and optimization procedures |
| Ollama Server Compatibility Issues | Medium | Medium | Comprehensive testing and fallback mechanisms |
| Network Connectivity Issues | Medium | High | Offline mode and local processing capabilities |
| Event Loop Conflicts in Async Workflows | Low | Low | nest_asyncio integration for hybrid async/sync support |
| LLM Classification Cost Escalation | Low | Low | Classification caching and batch processing optimization |
| Entity Resolution Scalability | Medium | Low | O(n²) algorithm limits to ~10K nodes; optimization for larger graphs |

### Business Risks

| Risk | Impact | Probability | Mitigation Strategy |
|------|--------|-------------|-------------------|
| AI Service Cost Escalation | High | Medium | Multi-provider support and usage monitoring |
| Regulatory Compliance Changes | Medium | Low | Modular design for compliance updates |
| Technology Obsolescence | Medium | Low | Regular technology assessments and migration planning |
| User Adoption Challenges | Medium | Medium | Comprehensive documentation and training materials |

### Operational Risks

| Risk | Impact | Probability | Mitigation Strategy |
|------|--------|-------------|-------------------|
| Data Loss During Processing | High | Low | Regular backups and transaction logging |
| Configuration Errors | High | Medium | Validation mechanisms and default fallbacks |
| Performance Bottlenecks | Medium | Medium | Monitoring and scaling capabilities |
| Security Vulnerabilities | High | Low | Regular security audits and updates |

### Key Assumptions

#### Technical Assumptions
- Python 3.13+ runtime environment availability
- Stable internet connectivity for cloud AI services
- Sufficient local hardware resources for Ollama processing
- Compatible PDF document formats and encodings
- Vector store persistence and recovery capabilities

#### Business Assumptions
- Users have access to required AI service accounts (Azure OpenAI, Poe)
- Local hardware meets Ollama processing requirements
- Acceptable performance for document processing workflows
- Willingness to use command-line interface

#### Environmental Assumptions
- Linux/Windows/macOS operating system compatibility
- File system permissions for document access
- Network security policies allow AI service access
- Local firewall configurations permit Ollama communication

---

## Product Overview

### Product Description

The PDF Agent is an intelligent research assistant designed to revolutionize how researchers, analysts, and knowledge workers interact with PDF document collections. By combining advanced AI technologies with intuitive natural language interfaces, the system transforms static document repositories into dynamic, conversational knowledge bases.

### Target Audience

#### Primary Users
- **Research Analysts**: Academic and industry researchers processing large document collections
- **Knowledge Workers**: Professionals requiring quick access to document insights
- **Data Scientists**: Users needing to analyze and synthesize information from PDFs
- **Legal Professionals**: Attorneys and paralegals researching case law and documents

#### Secondary Users
- **System Administrators**: IT personnel managing and maintaining the system
- **Content Managers**: Users responsible for document organization and indexing
- **Developers**: Technical users extending or integrating the system

### Value Proposition

#### User Benefits
- **Time Savings**: Reduce research time by 80% through intelligent document synthesis
- **Comprehensive Analysis**: Process entire document collections in minutes
- **Flexible Deployment**: Choose between cloud and local AI processing
- **Natural Interaction**: Conversational interface with context awareness
- **Privacy Control**: Local processing option for sensitive documents

#### Business Benefits
- **Cost Optimization**: Multi-provider support reduces AI service costs
- **Scalability**: Handle growing document collections efficiently
- **Reliability**: Redundant AI providers ensure system availability
- **Compliance**: Local processing meets data sovereignty requirements

### Product Context

The PDF Agent operates within the broader ecosystem of AI-powered research tools, positioning itself as a specialized solution for PDF document analysis. It integrates with existing document management workflows while providing advanced AI capabilities not available in traditional systems.

### Competitive Advantages

- **Multi-Provider AI**: Unique flexibility between cloud and local processing
- **Comprehensive Analysis**: Full collection processing capabilities
- **Memory Management**: Persistent conversation context
- **Open Architecture**: Extensible design for future enhancements
- **Privacy-First**: Local processing options for sensitive data

---

## Use Cases

### Primary Use Cases

#### UC-1: Document Collection Processing
**Actor**: Researcher/Analyst
**Preconditions**: PDF documents available, system configured
**Main Flow**:
1. User places PDF files in monitored directory
2. System automatically detects and processes new documents
3. Documents are parsed, chunked, and indexed with vector embeddings
4. User receives confirmation of successful processing
**Postconditions**: Documents available for search and analysis
**Alternative Flows**: Manual processing via command interface

#### UC-2: Intelligent Document Search
**Actor**: Researcher/Analyst
**Preconditions**: Document collection indexed
**Main Flow**:
1. User enters natural language query
2. System performs semantic search across document collection
3. Results ranked by relevance and presented with source citations
4. User can request additional context or clarification
**Postconditions**: User receives relevant information with sources
**Alternative Flows**: Enhanced search with conversation context

#### UC-3: Comprehensive Document Analysis
**Actor**: Researcher/Analyst
**Preconditions**: Large document collection indexed
**Main Flow**:
1. User requests comprehensive analysis of entire collection
2. System processes documents in batches
3. AI synthesizes findings across all documents
4. Comprehensive report generated with key insights
**Postconditions**: User receives holistic analysis of document collection

#### UC-4: Conversational Research Session
**Actor**: Researcher/Analyst
**Preconditions**: System initialized with document collection
**Main Flow**:
1. User starts interactive chat session
2. System maintains conversation context and history
3. User asks follow-up questions with context awareness
4. System provides relevant responses with source attribution
**Postconditions**: Natural research workflow with persistent context

#### UC-5: AI Provider Management
**Actor**: System Administrator/User
**Preconditions**: Multiple AI providers configured
**Main Flow**:
1. User checks current provider configuration
2. User selects preferred embedding provider (Azure/Ollama)
3. User selects preferred LLM provider (Poe/Ollama)
4. System validates configuration and confirms changes
**Postconditions**: System uses selected AI providers

#### UC-6: Async Paper Workflow
**Actor**: Researcher/Analyst
**Preconditions**: System configured with AI providers
**Main Flow**:
1. User issues natural language workflow command (e.g., "search, download, and process 50 papers about quantum computing")
2. System detects intent and extracts parameters (query, count, actions)
3. System confirms workflow with user
4. System executes async workflow: search → download → process
5. Process action builds knowledge graph without event loop conflicts
6. User receives summary of completed actions and processed documents
**Postconditions**: Papers downloaded, processed, and added to knowledge base
**Technical Notes**: Uses hybrid async/sync architecture with nest_asyncio for event loop compatibility

#### UC-7: Knowledge Graph Classification (NEW)
**Actor**: Researcher/System Administrator
**Preconditions**: Knowledge graph built from processed documents
**Main Flow**:
1. User checks current classification statistics via `graph stats`
2. User reviews ontology configuration via `ontology stats`
3. User runs hybrid classification via `graph reclassify-hybrid --dry-run` to preview
4. User confirms and executes `graph reclassify-hybrid`
5. System applies pattern/keyword classification (Phase 1+2)
6. System filters non-concepts (Phase 4 - timestamps, code artifacts, etc.)
7. System performs LLM-based classification on remaining unknowns (batched)
8. System caches LLM results for future runs
9. User reviews improved classification statistics
**Postconditions**: Graph classification improved from ~30% to target 55-60%
**Alternative Flows**: User can merge similar entities via `graph merge [threshold]` or normalize relationships via `graph normalize`

#### UC-8: Graph Entity Resolution (NEW)
**Actor**: Researcher/System Administrator
**Preconditions**: Knowledge graph contains potential duplicate entities
**Main Flow**:
1. User runs `graph merge 0.7` with default threshold
2. System calculates similarity between all entity pairs
3. System expands abbreviations (a2a → agent2agent, mcp → model context protocol)
4. System merges entities above similarity threshold
5. System preserves metadata (frequency, first_seen timestamps)
6. User receives report of merged entities
**Postconditions**: Duplicate entities consolidated, graph size reduced
**Technical Notes**: Uses Jaccard similarity + substring matching; configurable threshold 0.6-0.8

### Secondary Use Cases

#### UC-9: System Health Monitoring
**Actor**: System Administrator
**Preconditions**: System operational
**Main Flow**:
1. Administrator requests system statistics
2. System provides document count, performance metrics, and health status
3. Administrator reviews logs and error reports
**Postconditions**: Administrator informed of system status

#### UC-10: Memory Management
**Actor**: User
**Preconditions**: Active conversation session
**Main Flow**:
1. User reviews conversation history
2. User clears memory if needed
3. User starts new session with clean context
**Postconditions**: Conversation state managed appropriately

#### UC-11: Error Recovery
**Actor**: User/System Administrator
**Preconditions**: Error condition detected
**Main Flow**:
1. System detects and logs error condition
2. User receives clear error message with recovery suggestions
3. User follows recovery procedures or contacts support
**Postconditions**: Error resolved or escalated appropriately

---

## Requirements

### Functional Requirements

#### FR-1: Document Processing
**Description**: System shall automatically process PDF documents for search and analysis
**Requirements**:
- Parse PDF content with text extraction
- Generate vector embeddings for semantic search
- Support automatic and manual processing modes
- Handle various PDF formats and encodings
- Provide processing status and error reporting

#### FR-2: Search Functionality
**Description**: System shall provide multiple search modes for document queries
**Requirements**:
- Simple vector similarity search
- Enhanced search with conversation context
- Comprehensive analysis across entire collections
- Source attribution for all results
- Relevance ranking and scoring

#### FR-3: Conversational Interface
**Description**: System shall support natural language interaction with context awareness
**Requirements**:
- Maintain conversation history and context
- Support session management
- Provide context-aware responses
- Handle follow-up questions appropriately
- Clear memory when requested

#### FR-4: Multi-Provider AI Support
**Description**: System shall support multiple AI providers for flexibility
**Requirements**:
- Azure OpenAI for embeddings (optional)
- Poe API for LLM (optional)
- Ollama for local embeddings and LLM (optional)
- Runtime provider switching
- Graceful fallback mechanisms

#### FR-5: File System Integration
**Description**: System shall monitor and process documents automatically
**Requirements**:
- Directory monitoring for new PDFs
- Automatic processing triggers
- Duplicate detection and handling
- Batch processing capabilities
- File system permission handling

#### FR-6: Configuration Management
**Description**: System shall support flexible configuration options
**Requirements**:
- JSON-based configuration files
- Environment variable overrides
- Runtime provider selection
- Validation and error checking
- Default fallback values

#### FR-7: Error Handling and Recovery
**Description**: System shall handle errors gracefully with recovery options
**Requirements**:
- Comprehensive error logging
- User-friendly error messages
- Automatic retry mechanisms
- Recovery procedures documentation
- Graceful degradation options

#### FR-8: Async Workflow Support
**Description**: System shall support asynchronous workflows for paper search and processing
**Requirements**:
- Async paper search across multiple sources
- Concurrent download operations
- Hybrid async/sync architecture with event loop compatibility
- Knowledge graph building within async contexts (via nest_asyncio)
- Natural language workflow detection and parameter extraction
- Seamless integration of async orchestration with sync processing

#### FR-9: Knowledge Graph Classification (NEW)
**Description**: System shall provide multi-phase entity and relationship classification
**Requirements**:
- External YAML-based ontology configuration (48 ConceptTypes, 254+ keywords)
- Pattern-based classification (42 regex patterns: base, domain-specific, non-concept)
- Keyword-based classification with domain-specific categories
- LLM-based semantic classification with context awareness
- Non-concept filtering (timestamps, code artifacts, network elements)
- Classification result caching for cost optimization
- Batch processing for LLM classification with configurable batch size

#### FR-10: Graph Entity Management (NEW)
**Description**: System shall support entity resolution and relationship normalization
**Requirements**:
- Similarity-based duplicate entity detection
- Configurable similarity threshold (0.6-0.8)
- Abbreviation expansion and normalization
- Metadata preservation during entity merging
- Natural language relationship phrase mapping to standard types
- Support for 21 standard relationship types
- Graph visualization and query capabilities

#### FR-11: Graph Persistence and Retrieval (NEW)
**Description**: System shall persist and retrieve knowledge graph data efficiently
**Requirements**:
- JSON-based graph store persistence
- NetworkX in-memory graph representation
- Node attributes: type, frequency, first_seen, source_document
- Edge attributes: relationship_type, source_chunk
- Incremental graph updates
- Graph statistics and health reporting

### Data Requirements

#### DR-1: Document Storage
**Input**: PDF files
**Processing**: Text extraction, chunking, embedding generation
**Output**: Indexed documents with metadata
**Constraints**: Support for various PDF formats, size limits

#### DR-2: Vector Embeddings
**Input**: Document chunks (1024 tokens)
**Processing**: Embedding generation via selected provider
**Output**: 3072-dimensional vectors (Azure) or model-dependent (Ollama)
**Constraints**: Batch processing, rate limiting, error recovery

#### DR-3: Conversation History
**Input**: User messages and system responses
**Processing**: Context window management, session tracking
**Output**: Persistent chat history with metadata
**Constraints**: Token limits, session management, cleanup procedures

#### DR-4: System Configuration
**Input**: Configuration files and environment variables
**Processing**: Validation, hierarchy resolution, provider initialization
**Output**: Runtime configuration state
**Constraints**: Schema validation, secure credential handling

#### DR-5: Knowledge Graph Data (NEW)
**Input**: Entity triplets from LlamaIndex KG Extractor (15 per chunk)
**Processing**: Multi-phase classification (pattern/keyword/LLM), entity resolution, relationship normalization
**Output**: Classified entities and relationships with metadata
**Constraints**: NetworkX graph limits, O(n²) entity resolution scalability

#### DR-6: Ontology Configuration (NEW)
**Input**: YAML ontology file (`config/graph_ontology.yaml`)
**Processing**: Pattern compilation, keyword indexing, relationship mapping
**Output**: Loaded classification rules and validators
**Constraints**: YAML syntax validation, pattern regex validity

#### DR-7: Classification Cache (NEW)
**Input**: Node labels and LLM classification results
**Processing**: Cache lookup, result storage, statistics tracking
**Output**: Cached classifications and hit/miss statistics
**Constraints**: JSON serialization, automatic persistence every 10 entries

### Interface Requirements

#### IR-1: Command Line Interface
**Input**: Text commands and queries
**Output**: Formatted responses, status messages, error reports
**Constraints**: Rich formatting, interactive mode, command validation

#### IR-2: AI Provider Interfaces
**Input**: Text prompts, document chunks
**Output**: Embeddings, text completions, error responses
**Constraints**: API compatibility, rate limiting, authentication

#### IR-3: File System Interface
**Input**: PDF files, directory paths
**Output**: Processing status, indexed documents
**Constraints**: File permissions, monitoring reliability, error handling

#### IR-4: Graph Management Interface (NEW)
**Input**: Graph commands (`graph stats`, `graph reclassify-hybrid`, `graph merge`, `graph normalize`)
**Output**: Classification statistics, merged entities report, normalized relationships report
**Constraints**: Command validation, graph state consistency, large graph performance

#### IR-5: Ontology Management Interface (NEW)
**Input**: Ontology commands (`ontology stats`, `ontology show types`, `ontology show rels`, `ontology validate`)
**Output**: Rule statistics, concept/relationship type listings, validation results
**Constraints**: YAML parsing, configuration validation

---

## Configuration Steps

### Prerequisites

#### System Requirements
- **Operating System**: Linux, Windows, or macOS
- **Python Version**: 3.13 or higher
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space for documents and vector store
- **Network**: Internet connection for cloud AI services

#### Software Dependencies
- Python 3.13+ with pip
- Conda environment (recommended)
- Ollama (optional, for local AI processing)
- NetworkX 3.0+ (for knowledge graph)
- PyYAML 6.0+ (for ontology configuration)

### Installation Steps

#### Step 1: Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd pdf_agent

# Create conda environment
conda create -n pdf_agent python=3.13
conda activate pdf_agent

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: AI Provider Configuration

##### Option A: Cloud Providers (Azure + Poe)
```bash
# Configure system_config.json
{
  "azure_openai": {
    "api_key": "your-azure-key",
    "endpoint_url": "https://your-resource.openai.azure.com",
    "model_name": "text-embedding-3-large",
    "api_version": "2023-05-15",
    "embedding_dimension": 3072
  },
  "poe_service": {
    "api_key": "your-poe-key",
    "model_name": "Claude-Sonnet-4",
    "max_tokens": 4000,
    "temperature": 0.1
  },
  "agents": {
    "pdf_agent": {
      "config": {
        "embedding_provider": "azure",
        "llm_provider": "poe"
      }
    }
  }
}
```

##### Option B: Local Provider (Ollama)
```bash
# Install Ollama
# Visit: https://ollama.ai/download

# Pull required models
ollama pull nomic-embed-text
ollama pull llama2

# Configure system_config.json
{
  "ollama_service": {
    "base_url": "http://localhost:11434",
    "embedding_model": "nomic-embed-text",
    "model_name": "llama2",
    "max_tokens": 4000,
    "temperature": 0.1
  },
  "agents": {
    "pdf_agent": {
      "config": {
        "embedding_provider": "ollama",
        "llm_provider": "ollama"
      }
    }
  }
}
```

#### Step 3: Directory Setup
```bash
# Create required directories
mkdir -p storage/{vector_store,chat_history,graph_store}
mkdir -p logs
mkdir -p config

# Set permissions (Linux/macOS)
chmod 755 storage/
chmod 644 system_config.json

# Verify ontology configuration exists
ls -l config/graph_ontology.yaml
```

#### Step 4: System Testing
```bash
# Test Ollama (if using local provider)
python test_ollama.py

# Test system startup
python main.py
# Should display welcome message and stats

# Test knowledge graph classification
ontology stats
graph stats
```

#### Step 5: Document Processing Setup
```bash
# Create downloads directory for PDF monitoring
mkdir -p ~/Downloads/search_results

# Place test PDF files in the directory
# System will automatically process them

# Optionally build knowledge graph
python3 build_knowledge_graph.py

# Run hybrid classification (optional, improves classification to 55-60%)
graph reclassify-hybrid --dry-run  # Preview changes
graph reclassify-hybrid             # Execute classification
```

### Runtime Configuration

#### Provider Switching
```bash
# Check current providers
providers

# Switch embedding provider
set embedding azure
set embedding ollama

# Switch LLM provider
set llm poe
set llm ollama

# Note: Restart required for provider changes
```

#### Advanced Configuration
```bash
# Adjust processing parameters in system_config.json
{
  "chromadb": {
    "chunk_size": 1024,
    "chunk_overlap": 200
  },
  "memory": {
    "max_memory_tokens": 2000
  }
}
```

### Troubleshooting Configuration

#### Common Issues
1. **Import Errors**: Verify Python version and dependencies (including NetworkX, PyYAML)
2. **API Connection Failures**: Check credentials and network connectivity
3. **Ollama Connection Issues**: Verify Ollama server is running
4. **Permission Errors**: Check file system permissions
5. **Memory Issues**: Increase system RAM or reduce batch sizes
6. **Event Loop Errors**: Already resolved via nest_asyncio; ensure `nest-asyncio>=1.6.0` is installed
7. **YAML Parsing Errors**: Validate ontology configuration with `ontology validate`
8. **Graph Classification Performance**: For graphs >10K nodes, entity resolution may be slow (O(n²))
9. **LLM Classification Costs**: First-time hybrid classification costs ~$1-2; subsequent runs use cache ($0)

---

## Error Reporting

### Error Classification

#### System Errors
- **Configuration Errors**: Invalid settings, missing credentials
- **Initialization Failures**: Component startup issues
- **Resource Exhaustion**: Memory, disk space, or API limits
- **Network Issues**: Connectivity problems with AI services
- **Event Loop Conflicts**: Nested event loop issues (resolved via nest_asyncio)

#### Processing Errors
- **Document Parsing Failures**: Corrupted PDFs, unsupported formats
- **Embedding Generation Errors**: API failures, rate limiting
- **Index Corruption**: Vector store issues, data integrity problems
- **Search Failures**: Query processing errors, result formatting issues

#### User Interface Errors
- **Command Parsing Errors**: Invalid syntax, unknown commands
- **Session Management Issues**: Context corruption, memory limits
- **Display Problems**: Terminal compatibility, formatting issues

### Error Reporting Mechanisms

#### Logging Strategy
```python
# Structured logging with levels
log.info("Operation completed successfully")
log.warning("Non-critical issue detected")
log.error("Operation failed with error details")
log.exception("Exception with full traceback")
```

#### Error Message Format
```
[ERROR] Component: Message
Details: Additional context
Suggestion: Recommended action
Reference: Documentation link or command
```

#### User Notification Levels
- **Info**: Normal operation confirmations
- **Warning**: Potential issues that don't prevent operation
- **Error**: Operation failures requiring user action
- **Critical**: System stability threats requiring immediate attention

### Error Recovery Procedures

#### Automatic Recovery
- **API Retries**: Exponential backoff for transient failures
- **Connection Recovery**: Automatic reconnection to services
- **Resource Cleanup**: Memory and file handle management
- **Fallback Providers**: Switch to alternative AI providers

#### Manual Recovery
- **Configuration Reset**: Restore default settings
- **Data Recovery**: Rebuild indexes from source documents
- **Provider Switching**: Change AI providers for problematic services
- **System Restart**: Clean restart to resolve state issues

### Error Monitoring and Alerting

#### Health Checks
- Component status verification
- API connectivity testing
- Resource usage monitoring
- Performance threshold checking

#### Alert Conditions
- API failure rates > 5%
- Processing errors > 10%
- Memory usage > 90%
- Response times > 60 seconds

#### Log Analysis
- Error pattern identification
- Performance trend monitoring
- Usage anomaly detection
- Security event logging

### Error Documentation

#### User Guide Integration
- Error message explanations
- Troubleshooting procedures
- Recovery step-by-step guides
- Contact information for support

#### Developer Documentation
- Error code reference
- Exception handling patterns
- Debugging procedures
- Testing error scenarios

---

## Non-Functional Requirements

### Performance Requirements

#### Response Times
- **Simple Search**: < 15 seconds for 30 document chunks
- **Enhanced Search**: < 35 seconds with conversation context
- **Comprehensive Analysis**: < 5 minutes for 50 documents
- **Document Processing**: < 10 seconds per average PDF
- **System Startup**: < 30 seconds with index loading

#### Throughput
- **Document Processing**: 10 PDFs per minute
- **Search Queries**: 20 queries per minute
- **Concurrent Users**: 1 user (single-user design)
- **Batch Processing**: 10 documents per batch

#### Resource Utilization
- **Memory Usage**: < 2GB for typical operations
- **CPU Usage**: < 80% during processing peaks
- **Disk Space**: < 1GB per 100 PDFs (with embeddings)
- **Network Bandwidth**: < 10MB per hour for cloud providers

### Security Requirements

#### Data Protection
- **Credential Security**: Encrypted storage of API keys
- **Data Privacy**: Local processing option for sensitive documents
- **Access Control**: Single-user operation with file system permissions
- **Audit Logging**: Comprehensive activity logging without sensitive data

#### Network Security
- **API Communication**: HTTPS-only for cloud services
- **Certificate Validation**: SSL/TLS verification for all connections
- **Local Processing**: No external data transmission when using Ollama
- **Firewall Compliance**: Standard port usage (HTTP/HTTPS)

#### Application Security
- **Input Validation**: Sanitization of all user inputs
- **Error Handling**: No sensitive information in error messages
- **Dependency Security**: Regular updates of third-party libraries
- **Code Security**: Secure coding practices and regular security audits

### Usability Requirements

#### User Interface
- **Command Clarity**: Intuitive command structure with help system
- **Response Formatting**: Clear, readable output with rich formatting
- **Error Messages**: User-friendly error descriptions with actionable guidance
- **Progress Indicators**: Visual feedback for long-running operations

#### User Experience
- **Learning Curve**: < 30 minutes for basic operation proficiency
- **Task Completion**: Natural workflow for research tasks
- **Context Awareness**: Intuitive conversation flow with memory
- **Flexibility**: Multiple ways to accomplish common tasks

### Reliability Requirements

#### Availability
- **Uptime**: 99% availability during normal operations
- **Error Recovery**: Automatic recovery from transient failures
- **Graceful Degradation**: Continued operation with reduced functionality
- **Data Persistence**: No data loss during normal operation failures

#### Fault Tolerance
- **Provider Failover**: Automatic switching between AI providers
- **Data Integrity**: Transaction-like processing for critical operations
- **Resource Limits**: Prevention of resource exhaustion scenarios
- **Error Containment**: Isolation of component failures

### Maintainability Requirements

#### Code Quality
- **Modular Design**: Clear separation of concerns and responsibilities
- **Documentation**: Comprehensive inline and external documentation
- **Testing**: Unit and integration test coverage for critical paths
- **Version Control**: Proper branching and release management

#### Operational Requirements
- **Monitoring**: Comprehensive logging and health monitoring
- **Configuration**: Runtime configuration without code changes
- **Updates**: Rolling updates with backward compatibility
- **Supportability**: Clear error messages and troubleshooting guides

### Scalability Requirements

#### Data Scale
- **Document Capacity**: Support for 1000+ PDFs
- **Index Size**: Efficient handling of millions of document chunks
- **Search Performance**: Maintain response times with growing collections
- **Storage Efficiency**: Optimized storage usage and compression

#### Performance Scale
- **Concurrent Operations**: Support for multiple simultaneous operations
- **Batch Processing**: Efficient handling of large document sets
- **Memory Management**: Efficient resource usage under load
- **Network Optimization**: Minimize external API calls and bandwidth

### Compliance Requirements

#### Data Handling
- **Privacy Compliance**: Local processing options for sensitive data
- **Data Retention**: Configurable cleanup and retention policies
- **Audit Trails**: Comprehensive logging for compliance verification
- **Data Sovereignty**: Support for local data processing requirements

#### Security Standards
- **Secure Defaults**: Secure configuration out-of-the-box
- **Access Logging**: Detailed access and operation logging
- **Vulnerability Management**: Regular security updates and patches
- **Incident Response**: Defined procedures for security incidents

---

## Document Approval

### Review Status
- **Technical Review**: ✅ Completed
- **Security Review**: ✅ Completed
- **User Experience Review**: ✅ Completed
- **Architecture Review**: ✅ Completed

### Approval Signatures
- **Product Owner**: ____________________ Date: ____________
- **Technical Lead**: ____________________ Date: ____________
- **Security Officer**: ____________________ Date: ____________
- **Architecture Lead**: ____________________ Date: ____________

### Document Version History
- **v1.0** (October 23, 2025): Initial functional design document with Ollama integration
- **v1.1** (October 27, 2025): Added LlamaIndex migration notes, Azure/Poe and Ollama provider stabilization, memory & retrieval fixes, and Knowledge Graph integration
- **v1.2** (October 30, 2025): Added async workflow support with nest_asyncio integration, hybrid async/sync architecture, event loop conflict resolution, new use case UC-6 for async paper workflows, and updated functional requirements FR-8
- **v1.3** (October 31, 2025): Added knowledge graph classification enhancements with 4-phase strategy (YAML config, domain patterns, hybrid LLM classification). Added use cases UC-7 (Knowledge Graph Classification) and UC-8 (Graph Entity Resolution). Added functional requirements FR-9 (Knowledge Graph Classification), FR-10 (Graph Entity Management), and FR-11 (Graph Persistence). Added data requirements DR-5, DR-6, DR-7 for graph data, ontology, and cache. Added interface requirements IR-4 and IR-5 for graph and ontology management. Updated configuration steps, troubleshooting, and technical risks.

---

*This Functional Design Document serves as the blueprint for the PDF Agent system implementation. All development activities should align with the requirements and specifications outlined in this document.*