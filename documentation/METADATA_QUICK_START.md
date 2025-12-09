# Metadata Database - Quick Start Guide

## Overview

The PDF Agent now includes a SQLite3 metadata database that automatically tracks all metadata for documents, embeddings, and knowledge graph nodes/relationships.

## Location

```
/storage/metadata.db
```

## What Gets Automatically Saved

### When You Process PDFs

Every time you process a PDF, the system automatically saves:
- Document ID (UUID)
- File name, path, and hash
- Title, authors, keywords (if available)
- Parse timestamp
- Number of chunks created

### When Embeddings Are Created

For every chunk that gets embedded:
- Embedding ID (node ID)
- Link to parent document
- Chunk index and text
- Vector dimension
- Embedding provider and model
- ChromaDB collection name

### When Knowledge Graph Is Built

For every node in the knowledge graph:
- Node ID (hash-based)
- Node label and type
- Classification method (pattern/keyword/LLM/hybrid)
- Classification confidence
- Frequency (how often it appears)
- First seen timestamp

For every relationship:
- Relationship ID
- Source and target nodes
- Relationship type and normalized type
- Weight
- Source documents

## How to Use It

### Check Statistics

```bash
python tools/metadata_db_cli.py stats
```

Output:
```
📊 Overall Statistics:
  Documents:     5
  Embeddings:    127
  Graph Nodes:   342
  Relationships: 856
  Database Size: 2.5 MB

🔍 Node Statistics:
  Total Nodes: 342
  By Type (Top 10):
    machine_learning               45
    neural_network                 38
    algorithm                      27
    ...
```

### List Documents

```bash
# List all documents
python tools/metadata_db_cli.py documents

# Limit results
python tools/metadata_db_cli.py documents --limit 5
```

### View Document Details

```bash
python tools/metadata_db_cli.py document <document_id>
```

Shows:
- Full document metadata
- All embeddings for the document
- Source papers linked to graph nodes

### Browse Graph Nodes

```bash
# List all nodes
python tools/metadata_db_cli.py nodes --limit 20

# Filter by type
python tools/metadata_db_cli.py nodes --type machine_learning

# Show node details
python tools/metadata_db_cli.py node <node_id>
```

### Explore Relationships

```bash
# List all relationships
python tools/metadata_db_cli.py relationships --limit 10

# Filter by type
python tools/metadata_db_cli.py relationships --type "uses"
```

### Export Data

```bash
# Export entire database to JSON
python tools/metadata_db_cli.py export metadata_export.json
```

## Python API

### Get Database Instance

```python
from core.metadata_db import get_metadata_db

db = get_metadata_db()
```

### Query Documents

```python
# Get document by ID
doc = db.get_document("doc-123")
print(doc['title'])
print(doc['authors'])  # JSON array

# Check if file already processed
existing = db.get_document_by_hash("abc123...")
if existing:
    print("Already processed!")

# List all documents
docs = db.list_documents(limit=10)
for doc in docs:
    print(f"{doc['file_name']}: {doc['chunk_count']} chunks")
```

### Query Embeddings

```python
# Get all embeddings for a document
embeddings = db.get_document_embeddings("doc-123")
for emb in embeddings:
    print(f"Chunk {emb['chunk_index']}: {emb['chunk_text'][:50]}...")

# Get single embedding
emb = db.get_embedding("emb-456")
print(f"Provider: {emb['embedding_provider']}")
print(f"Model: {emb['embedding_model']}")
print(f"Dimension: {emb['vector_dimension']}")
```

### Query Graph Nodes

```python
# Get nodes by type
ml_nodes = db.get_nodes_by_type("machine_learning")
print(f"Found {len(ml_nodes)} ML nodes")

# Get node details
node = db.get_graph_node("node-789")
print(f"{node['label']}: {node['node_type']}")
print(f"Classified by: {node['classification_method']}")
print(f"Frequency: {node['frequency']}")

# Get node relationships
rels = db.get_node_relationships("node-789", direction='outgoing')
for rel in rels:
    print(f"→ {rel['relationship_type']} → {rel['target_node_id']}")
```

### Query Relationships

```python
# Get relationship
rel = db.get_relationship("rel-abc")
print(f"{rel['source_node_id']} --[{rel['relationship_type']}]--> {rel['target_node_id']}")

# Get statistics
rel_stats = db.get_relationship_statistics()
print(f"Total relationships: {rel_stats['total_relationships']}")
print(f"Most common type: {max(rel_stats['by_type'].items(), key=lambda x: x[1])}")
```

### Get Statistics

```python
# Overall statistics
stats = db.get_database_statistics()
print(f"Documents: {stats['documents']}")
print(f"Embeddings: {stats['embeddings']}")
print(f"Nodes: {stats['graph_nodes']}")
print(f"Relationships: {stats['relationships']}")
print(f"Size: {stats['database_size_bytes'] / 1024 / 1024:.2f} MB")

# Node statistics
node_stats = db.get_node_statistics()
print("Node type distribution:")
for node_type, count in sorted(node_stats['by_type'].items(), key=lambda x: x[1], reverse=True):
    print(f"  {node_type}: {count}")

# Classification method distribution
for method, count in node_stats['by_classification_method'].items():
    print(f"  {method}: {count}")
```

### Link Nodes to Embeddings

```python
# Link a node to an embedding (for semantic search)
db.link_node_to_embedding(
    node_id="node-789",
    embedding_id="emb-456",
    relevance_score=0.85
)

# Get all embeddings for a node
node_embs = db.get_node_embeddings("node-789")
for emb in node_embs:
    print(f"Score {emb['relevance_score']}: {emb['chunk_text'][:50]}...")

# Get all nodes for an embedding
emb_nodes = db.get_embedding_nodes("emb-456")
for node in emb_nodes:
    print(f"{node['label']} ({node['node_type']}) - Score: {node['relevance_score']}")
```

## Common Queries

### Find Documents by Author

```python
docs = db.list_documents()
for doc in docs:
    authors = json.loads(doc['authors']) if doc['authors'] else []
    if 'Smith' in authors:
        print(f"{doc['title']} - {', '.join(authors)}")
```

### Find Most Frequent Nodes

```python
with db.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT label, node_type, frequency
        FROM graph_nodes
        ORDER BY frequency DESC
        LIMIT 10
    """)
    for row in cursor.fetchall():
        print(f"{row['label']} ({row['node_type']}): {row['frequency']}")
```

### Find Nodes by Classification Method

```python
with db.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) as count, classification_method
        FROM graph_nodes
        GROUP BY classification_method
    """)
    for row in cursor.fetchall():
        print(f"{row['classification_method']}: {row['count']}")
```

### Find Documents with Most Chunks

```python
with db.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT file_name, chunk_count
        FROM documents
        ORDER BY chunk_count DESC
        LIMIT 10
    """)
    for row in cursor.fetchall():
        print(f"{row['file_name']}: {row['chunk_count']} chunks")
```

## Maintenance

### Backup Database

```bash
# Simple copy
cp storage/metadata.db storage/metadata.db.backup

# SQLite backup
sqlite3 storage/metadata.db ".backup storage/metadata.db.backup"
```

### Optimize Database

```python
from core.metadata_db import get_metadata_db

db = get_metadata_db()
db.vacuum()  # Rebuild and optimize
```

### Inspect with SQLite CLI

```bash
sqlite3 storage/metadata.db

# Useful commands:
.tables                           # List all tables
.schema documents                 # Show table schema
SELECT COUNT(*) FROM documents;   # Count documents
SELECT * FROM graph_nodes LIMIT 10;  # Browse nodes
.quit                             # Exit
```

## Integration Notes

### Automatic Operation

The metadata database is **automatically populated** during normal system operation:

1. When you run `process <folder>` - Documents and embeddings are saved
2. When knowledge graph is built - Nodes and relationships are saved
3. When graph is synced - Metadata is updated

### No Manual Intervention Required

You don't need to explicitly call database methods in normal usage. The integration is transparent:

- `PDFParser` automatically saves document metadata
- `GraphManager` automatically saves node metadata
- Both components link embeddings appropriately

### Access When Needed

Use the database when you need to:
- Query metadata across the system
- Generate reports and statistics
- Debug processing issues
- Track provenance of nodes/embeddings
- Export data for external analysis

## Troubleshooting

### Database Locked

If you get "database is locked" errors:
- Ensure only one process accesses the database at a time
- The code uses context managers to prevent this
- If it persists, restart the application

### Missing Data

If expected data is missing:
- Check logs for errors during processing
- Verify integration is enabled in both PDFParser and GraphManager
- Run test suite: `python tests/test_metadata_db.py`

### Large Database Size

If database grows too large:
- Run `db.vacuum()` to reclaim space
- Consider archiving old documents
- Check for duplicate entries

## See Also

- [METADATA_DATABASE.md](METADATA_DATABASE.md) - Full documentation
- [METADATA_IMPLEMENTATION_SUMMARY.md](METADATA_IMPLEMENTATION_SUMMARY.md) - Implementation details
- [core/metadata_db.py](../core/metadata_db.py) - Source code
- [tools/metadata_db_cli.py](../tools/metadata_db_cli.py) - CLI tool
