# Metadata Database Documentation

## Overview

The PDF Agent system includes a SQLite3 metadata database that stores comprehensive metadata for documents, embeddings, and knowledge graph nodes/relationships. This database provides a unified interface for tracking and querying all metadata across the system.

## Database Location

The SQLite database is stored at:
```
/storage/metadata.db
```

## Architecture

The metadata database serves as the central metadata repository that bridges:
1. **Document Processing** - PDF parsing and text extraction
2. **Vector Embeddings** - Chunk-level embeddings in ChromaDB
3. **Knowledge Graph** - Nodes and relationships in NetworkX/SimpleGraphStore

### Integration Points

```
┌─────────────────┐
│   PDF Parser    │
│  (pdf_parser.py)│
└────────┬────────┘
         │
         ├──> Documents Table
         └──> Embeddings Table
              │
              v
┌─────────────────────────┐
│    Vector Store         │
│    (ChromaDB)           │
└─────────────────────────┘

┌─────────────────┐
│  Graph Manager  │
│(graph_manager.py)│
└────────┬────────┘
         │
         ├──> Graph Nodes Table
         ├──> Relationships Table
         └──> Node-Embedding Links
              │
              v
┌─────────────────────────┐
│   Knowledge Graph       │
│   (NetworkX/SimpleStore)│
└─────────────────────────┘
```

## Database Schema

### Tables

#### 1. `documents`
Stores core paper/PDF metadata.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment primary key |
| document_id | TEXT UNIQUE | UUID for the document |
| file_name | TEXT | Original filename |
| file_path | TEXT | Absolute file path |
| file_hash | TEXT UNIQUE | MD5 hash for deduplication |
| file_type | TEXT | File type (default: 'pdf') |
| title | TEXT | Document title |
| authors | TEXT | JSON array of authors |
| publication_date | TEXT | Publication date |
| abstract | TEXT | Document abstract |
| keywords | TEXT | JSON array of keywords |
| doi | TEXT | Digital Object Identifier |
| url | TEXT | Source URL |
| parsed_date | TEXT | When document was parsed |
| chunk_count | INTEGER | Number of chunks created |
| metadata_json | TEXT | Additional metadata as JSON |
| created_at | TEXT | Record creation timestamp |
| updated_at | TEXT | Last update timestamp |

#### 2. `embeddings`
Stores references to vector embeddings in ChromaDB.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment primary key |
| embedding_id | TEXT UNIQUE | UUID/node_id for embedding |
| document_id | TEXT FK | Reference to documents table |
| chunk_index | INTEGER | Index of chunk within document |
| chunk_text | TEXT | Original chunk text |
| chunk_size | INTEGER | Size of chunk in characters |
| vector_dimension | INTEGER | Embedding vector dimension |
| embedding_provider | TEXT | Provider (azure/ollama) |
| embedding_model | TEXT | Model name |
| vector_store_collection | TEXT | ChromaDB collection name |
| metadata_json | TEXT | Additional metadata as JSON |
| created_at | TEXT | Record creation timestamp |

#### 3. `graph_nodes`
Stores knowledge graph node metadata.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment primary key |
| node_id | TEXT UNIQUE | SHA256 hash of node label (16 chars) |
| label | TEXT | Node label/name |
| node_type | TEXT | Concept type (from ontology) |
| classification_method | TEXT | How classified (pattern/keyword/llm/hybrid) |
| classification_confidence | REAL | Confidence score (0-1) |
| frequency | INTEGER | How often node appears |
| first_seen | TEXT | When first encountered |
| last_updated | TEXT | Last update timestamp |
| source_documents | TEXT | JSON array of document_ids |
| properties_json | TEXT | Additional properties as JSON |
| created_at | TEXT | Record creation timestamp |
| updated_at | TEXT | Last update timestamp |

#### 4. `relationships`
Stores knowledge graph edges/relationships.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment primary key |
| relationship_id | TEXT UNIQUE | SHA256 hash of relationship |
| source_node_id | TEXT FK | Source node reference |
| target_node_id | TEXT FK | Target node reference |
| relationship_type | TEXT | Original relationship type |
| normalized_type | TEXT | Normalized relationship type |
| weight | REAL | Edge weight (default: 1.0) |
| source_documents | TEXT | JSON array of document_ids |
| properties_json | TEXT | Additional properties as JSON |
| created_at | TEXT | Record creation timestamp |
| updated_at | TEXT | Last update timestamp |

#### 5. `document_embeddings`
Many-to-many mapping between documents and embeddings.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment primary key |
| document_id | TEXT FK | Reference to documents |
| embedding_id | TEXT FK | Reference to embeddings |

#### 6. `node_embeddings`
Links knowledge graph nodes to embeddings for semantic search.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment primary key |
| node_id | TEXT FK | Reference to graph_nodes |
| embedding_id | TEXT FK | Reference to embeddings |
| relevance_score | REAL | Relevance score (optional) |

### Indexes

For optimal query performance, the following indexes are created:

- `idx_documents_file_hash` - Fast duplicate detection
- `idx_embeddings_document_id` - Quick document→embeddings lookup
- `idx_embeddings_provider` - Filter by embedding provider
- `idx_graph_nodes_label` - Search nodes by label
- `idx_graph_nodes_type` - Filter nodes by type
- `idx_relationships_source` - Outgoing relationships
- `idx_relationships_target` - Incoming relationships
- `idx_relationships_type` - Filter by relationship type

## Usage

### Python API

```python
from core.metadata_db import get_metadata_db

# Get database instance
db = get_metadata_db()

# Insert a document
db.insert_document(
    document_id="doc-123",
    file_name="paper.pdf",
    file_path="/path/to/paper.pdf",
    file_hash="abc123...",
    parsed_date="2025-10-31T12:00:00",
    title="Research Paper Title",
    authors=["Author 1", "Author 2"]
)

# Query document
doc = db.get_document("doc-123")
print(doc['title'])

# Get document embeddings
embeddings = db.get_document_embeddings("doc-123")
for emb in embeddings:
    print(f"Chunk {emb['chunk_index']}: {emb['chunk_text'][:50]}...")

# Insert graph node
db.insert_graph_node(
    node_id="node-456",
    label="machine learning",
    node_type="machine_learning",
    classification_method="keyword",
    frequency=5
)

# Query nodes by type
ml_nodes = db.get_nodes_by_type("machine_learning")

# Get statistics
stats = db.get_database_statistics()
print(f"Total documents: {stats['documents']}")
print(f"Total nodes: {stats['graph_nodes']}")
```

### Command Line Interface

The `metadata_db_cli.py` tool provides command-line access:

```bash
# Show database statistics
python tools/metadata_db_cli.py stats

# List all documents
python tools/metadata_db_cli.py documents --limit 10

# Show document details
python tools/metadata_db_cli.py document <document_id>

# List graph nodes
python tools/metadata_db_cli.py nodes --type machine_learning --limit 20

# Show node details
python tools/metadata_db_cli.py node <node_id>

# List relationships
python tools/metadata_db_cli.py relationships --type "uses" --limit 10

# Export database to JSON
python tools/metadata_db_cli.py export output.json
```

## Integration with System Components

### PDF Parser Integration

When processing PDFs, the `PDFParser` automatically:
1. Creates document record with file metadata
2. Creates embedding records for each chunk
3. Links embeddings to documents
4. Updates chunk counts

```python
# In pdf_parser.py
from core.metadata_db import get_metadata_db

class PDFParser:
    def __init__(self):
        self.metadata_db = get_metadata_db()
    
    def parse_pdf(self, file_path):
        # ... parsing logic ...
        
        # Save metadata
        self.metadata_db.insert_document(
            document_id=document_id,
            file_name=file_path.name,
            file_path=str(file_path),
            file_hash=file_hash,
            parsed_date=datetime.now().isoformat()
        )
```

### Graph Manager Integration

When building knowledge graphs, the `GraphManager` automatically:
1. Creates node records with classification metadata
2. Creates relationship records
3. Links nodes to embeddings for semantic search

```python
# In graph_manager.py
from core.metadata_db import get_metadata_db

class GraphManager:
    def __init__(self):
        self.metadata_db = get_metadata_db()
    
    def _sync_to_networkx(self):
        # ... graph sync logic ...
        
        # Save nodes
        self.metadata_db.insert_graph_node(
            node_id=node_id,
            label=node_label,
            node_type=classification,
            classification_method='pattern',
            frequency=freq
        )
        
        # Save relationships
        self.metadata_db.insert_relationship(
            relationship_id=rel_id,
            source_node_id=source,
            target_node_id=target,
            relationship_type=rel_type
        )
```

## Query Examples

### Find all embeddings for a document
```python
embeddings = db.get_document_embeddings("doc-123")
```

### Find all nodes of a specific type
```python
algorithm_nodes = db.get_nodes_by_type("algorithm")
```

### Find all relationships involving a node
```python
# Outgoing only
outgoing = db.get_node_relationships("node-456", direction='outgoing')

# Incoming only
incoming = db.get_node_relationships("node-456", direction='incoming')

# Both directions
all_rels = db.get_node_relationships("node-456", direction='both')
```

### Find documents by hash (deduplication)
```python
existing = db.get_document_by_hash("abc123...")
if existing:
    print("Document already processed")
```

### Get comprehensive statistics
```python
# Overall stats
overall = db.get_database_statistics()

# Node statistics with type breakdown
node_stats = db.get_node_statistics()
print(f"Unknown nodes: {node_stats['by_type'].get('unknown', 0)}")

# Relationship statistics
rel_stats = db.get_relationship_statistics()
print(f"Most common relationship: {max(rel_stats['by_type'].items(), key=lambda x: x[1])}")
```

## Maintenance

### Database Optimization
```python
db = get_metadata_db()
db.vacuum()  # Rebuild database to reclaim space and optimize
```

### Backup
```bash
# Simple file copy
cp storage/metadata.db storage/metadata.db.backup

# SQLite backup command
sqlite3 storage/metadata.db ".backup storage/metadata.db.backup"
```

### Inspection
```bash
# Open database in SQLite CLI
sqlite3 storage/metadata.db

# Useful commands:
.tables                    # List all tables
.schema documents         # Show table schema
SELECT COUNT(*) FROM documents;
SELECT * FROM graph_nodes LIMIT 10;
```

## Performance Considerations

- **Indexes**: All key columns are indexed for fast queries
- **Batch Operations**: Use transactions for bulk inserts
- **Connection Pooling**: Context manager ensures proper connection handling
- **VACUUM**: Run periodically to optimize database file

## Future Enhancements

Potential improvements:
1. Full-text search on document content using FTS5
2. Temporal queries for tracking changes over time
3. Graph traversal queries using recursive CTEs
4. Embedding similarity search integration
5. Automated backup and archival strategies
6. Multi-user access with connection pooling
7. Integration with external metadata sources (CrossRef, arXiv API)

## Troubleshooting

### Database locked errors
- Ensure only one process accesses the database at a time
- Use context managers for proper connection handling

### Large database size
- Run `db.vacuum()` to reclaim space
- Consider archiving old documents
- Check for duplicate entries

### Missing metadata
- Verify integration points are enabled
- Check logs for errors during document processing
- Ensure all components use `get_metadata_db()` singleton

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture overview
- [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md) - Technical details
- [core/metadata_db.py](../core/metadata_db.py) - Implementation
- [tools/metadata_db_cli.py](../tools/metadata_db_cli.py) - CLI tool
