"""
SQLite3 Metadata Database Manager

This module manages a SQLite3 database for storing document and node metadata.
The metadata is referenced by:
1. Paper embeddings (via document_id and embedding_id)
2. Knowledge graph nodes (via node_id)

Database Schema:
- documents: Core paper/PDF metadata
- embeddings: Vector embedding references
- graph_nodes: Knowledge graph node metadata
- relationships: Knowledge graph relationships
- document_embeddings: Many-to-many mapping between documents and embeddings
- node_embeddings: Mapping between graph nodes and embeddings
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
from contextlib import contextmanager

from config import settings
from utils.logger import log


class MetadataDatabase:
    """Manage SQLite3 database for paper and graph metadata"""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize metadata database.
        
        Args:
            db_path: Path to SQLite database file. Defaults to storage/metadata.db
        """
        if db_path is None:
            db_path = settings.STORAGE_PATH / "metadata.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        self._initialize_schema()
        log.info(f"MetadataDatabase initialized at: {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _initialize_schema(self):
        """Create database schema if it doesn't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Documents table - stores core paper/PDF metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT UNIQUE NOT NULL,
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT UNIQUE NOT NULL,
                    file_type TEXT DEFAULT 'pdf',
                    title TEXT,
                    authors TEXT,  -- JSON array of authors
                    publication_date TEXT,
                    abstract TEXT,
                    keywords TEXT,  -- JSON array of keywords
                    doi TEXT,
                    url TEXT,
                    parsed_date TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    metadata_json TEXT,  -- Additional metadata as JSON
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Embeddings table - stores references to vector embeddings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    embedding_id TEXT UNIQUE NOT NULL,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER,
                    chunk_text TEXT,
                    chunk_size INTEGER,
                    vector_dimension INTEGER,
                    embedding_provider TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    vector_store_collection TEXT,
                    metadata_json TEXT,  -- Additional metadata as JSON
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(document_id)
                )
            """)
            
            # Graph nodes table - stores knowledge graph node metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT UNIQUE NOT NULL,
                    label TEXT NOT NULL,
                    node_type TEXT DEFAULT 'unknown',
                    classification_method TEXT,  -- pattern, keyword, llm, hybrid
                    classification_confidence REAL,
                    frequency INTEGER DEFAULT 1,
                    first_seen TEXT,
                    last_updated TEXT,
                    source_documents TEXT,  -- JSON array of document_ids
                    properties_json TEXT,  -- Additional properties as JSON
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Relationships table - stores knowledge graph edges
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    relationship_id TEXT UNIQUE NOT NULL,
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    normalized_type TEXT,
                    weight REAL DEFAULT 1.0,
                    source_documents TEXT,  -- JSON array of document_ids
                    properties_json TEXT,  -- Additional properties as JSON
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_node_id) REFERENCES graph_nodes(node_id),
                    FOREIGN KEY (target_node_id) REFERENCES graph_nodes(node_id)
                )
            """)
            
            # Document-Embeddings mapping table (many-to-many)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    embedding_id TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(document_id),
                    FOREIGN KEY (embedding_id) REFERENCES embeddings(embedding_id),
                    UNIQUE(document_id, embedding_id)
                )
            """)
            
            # Node-Embeddings mapping table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS node_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    embedding_id TEXT NOT NULL,
                    relevance_score REAL,
                    FOREIGN KEY (node_id) REFERENCES graph_nodes(node_id),
                    FOREIGN KEY (embedding_id) REFERENCES embeddings(embedding_id),
                    UNIQUE(node_id, embedding_id)
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_file_hash 
                ON documents(file_hash)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_document_id 
                ON embeddings(document_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_provider 
                ON embeddings(embedding_provider)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_graph_nodes_label 
                ON graph_nodes(label)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_graph_nodes_type 
                ON graph_nodes(node_type)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationships_source 
                ON relationships(source_node_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationships_target 
                ON relationships(target_node_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationships_type 
                ON relationships(relationship_type)
            """)
            
            conn.commit()
            log.info("✓ Database schema initialized")
    
    # ==================== Document Operations ====================
    
    def insert_document(
        self,
        document_id: str,
        file_name: str,
        file_path: str,
        file_hash: str,
        parsed_date: str,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        publication_date: Optional[str] = None,
        abstract: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        doi: Optional[str] = None,
        url: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Insert a new document record.
        
        Returns:
            Database row ID of inserted document
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO documents (
                    document_id, file_name, file_path, file_hash, 
                    title, authors, publication_date, abstract, 
                    keywords, doi, url, parsed_date, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                document_id,
                file_name,
                file_path,
                file_hash,
                title,
                json.dumps(authors) if authors else None,
                publication_date,
                abstract,
                json.dumps(keywords) if keywords else None,
                doi,
                url,
                parsed_date,
                json.dumps(metadata) if metadata else None
            ))
            
            log.debug(f"Inserted document: {document_id}")
            return cursor.lastrowid
    
    def get_document(self, document_id: str) -> Optional[Dict]:
        """Get document by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    def get_document_by_hash(self, file_hash: str) -> Optional[Dict]:
        """Get document by file hash"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM documents WHERE file_hash = ?",
                (file_hash,)
            )
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    def update_document_chunk_count(self, document_id: str, chunk_count: int):
        """Update chunk count for a document"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE documents 
                SET chunk_count = ?, updated_at = CURRENT_TIMESTAMP
                WHERE document_id = ?
            """, (chunk_count, document_id))
    
    def list_documents(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict]:
        """List all documents with pagination"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM documents ORDER BY created_at DESC"
            if limit:
                query += f" LIMIT {limit} OFFSET {offset}"
            
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== Embedding Operations ====================
    
    def insert_embedding(
        self,
        embedding_id: str,
        document_id: str,
        chunk_index: int,
        chunk_text: str,
        embedding_provider: str,
        embedding_model: str,
        chunk_size: Optional[int] = None,
        vector_dimension: Optional[int] = None,
        vector_store_collection: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Insert a new embedding record.
        
        Returns:
            Database row ID of inserted embedding
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings (
                    embedding_id, document_id, chunk_index, chunk_text,
                    chunk_size, vector_dimension, embedding_provider,
                    embedding_model, vector_store_collection, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                embedding_id,
                document_id,
                chunk_index,
                chunk_text,
                chunk_size,
                vector_dimension,
                embedding_provider,
                embedding_model,
                vector_store_collection,
                json.dumps(metadata) if metadata else None
            ))
            
            # Also create document-embedding mapping
            cursor.execute("""
                INSERT OR IGNORE INTO document_embeddings (
                    document_id, embedding_id
                ) VALUES (?, ?)
            """, (document_id, embedding_id))
            
            log.debug(f"Inserted embedding: {embedding_id}")
            return cursor.lastrowid
    
    def get_embedding(self, embedding_id: str) -> Optional[Dict]:
        """Get embedding by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM embeddings WHERE embedding_id = ?",
                (embedding_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    def get_document_embeddings(self, document_id: str) -> List[Dict]:
        """Get all embeddings for a document"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM embeddings WHERE document_id = ? ORDER BY chunk_index",
                (document_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== Graph Node Operations ====================
    
    def insert_graph_node(
        self,
        node_id: str,
        label: str,
        node_type: str = 'unknown',
        classification_method: Optional[str] = None,
        classification_confidence: Optional[float] = None,
        frequency: int = 1,
        first_seen: Optional[str] = None,
        source_documents: Optional[List[str]] = None,
        properties: Optional[Dict] = None
    ) -> int:
        """
        Insert or update a graph node record.
        
        Returns:
            Database row ID of inserted/updated node
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if first_seen is None:
                first_seen = datetime.now().isoformat()
            
            cursor.execute("""
                INSERT OR REPLACE INTO graph_nodes (
                    node_id, label, node_type, classification_method,
                    classification_confidence, frequency, first_seen,
                    last_updated, source_documents, properties_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                node_id,
                label,
                node_type,
                classification_method,
                classification_confidence,
                frequency,
                first_seen,
                datetime.now().isoformat(),
                json.dumps(source_documents) if source_documents else None,
                json.dumps(properties) if properties else None
            ))
            
            log.debug(f"Inserted graph node: {node_id} ({node_type})")
            return cursor.lastrowid
    
    def get_graph_node(self, node_id: str) -> Optional[Dict]:
        """Get graph node by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM graph_nodes WHERE node_id = ?",
                (node_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    def update_node_classification(
        self,
        node_id: str,
        node_type: str,
        classification_method: str,
        confidence: Optional[float] = None
    ):
        """Update node classification"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE graph_nodes 
                SET node_type = ?,
                    classification_method = ?,
                    classification_confidence = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE node_id = ?
            """, (node_type, classification_method, confidence, node_id))
    
    def increment_node_frequency(self, node_id: str):
        """Increment node frequency counter"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE graph_nodes 
                SET frequency = frequency + 1,
                    last_updated = CURRENT_TIMESTAMP
                WHERE node_id = ?
            """, (node_id,))
    
    def get_nodes_by_type(self, node_type: str) -> List[Dict]:
        """Get all nodes of a specific type"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM graph_nodes WHERE node_type = ? ORDER BY frequency DESC",
                (node_type,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_node_statistics(self) -> Dict:
        """Get statistics about graph nodes"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total nodes
            cursor.execute("SELECT COUNT(*) as total FROM graph_nodes")
            total = cursor.fetchone()['total']
            
            # Nodes by type
            cursor.execute("""
                SELECT node_type, COUNT(*) as count
                FROM graph_nodes
                GROUP BY node_type
                ORDER BY count DESC
            """)
            by_type = {row['node_type']: row['count'] for row in cursor.fetchall()}
            
            # Classification methods
            cursor.execute("""
                SELECT classification_method, COUNT(*) as count
                FROM graph_nodes
                WHERE classification_method IS NOT NULL
                GROUP BY classification_method
            """)
            by_method = {row['classification_method']: row['count'] for row in cursor.fetchall()}
            
            return {
                'total_nodes': total,
                'by_type': by_type,
                'by_classification_method': by_method
            }
    
    # ==================== Relationship Operations ====================
    
    def insert_relationship(
        self,
        relationship_id: str,
        source_node_id: str,
        target_node_id: str,
        relationship_type: str,
        normalized_type: Optional[str] = None,
        weight: float = 1.0,
        source_documents: Optional[List[str]] = None,
        properties: Optional[Dict] = None
    ) -> int:
        """
        Insert or update a relationship record.
        
        Returns:
            Database row ID of inserted/updated relationship
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO relationships (
                    relationship_id, source_node_id, target_node_id,
                    relationship_type, normalized_type, weight,
                    source_documents, properties_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                relationship_id,
                source_node_id,
                target_node_id,
                relationship_type,
                normalized_type or relationship_type,
                weight,
                json.dumps(source_documents) if source_documents else None,
                json.dumps(properties) if properties else None
            ))
            
            log.debug(f"Inserted relationship: {source_node_id} --[{relationship_type}]--> {target_node_id}")
            return cursor.lastrowid
    
    def get_relationship(self, relationship_id: str) -> Optional[Dict]:
        """Get relationship by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM relationships WHERE relationship_id = ?",
                (relationship_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    def get_node_relationships(
        self,
        node_id: str,
        direction: str = 'both'  # 'outgoing', 'incoming', 'both'
    ) -> List[Dict]:
        """Get all relationships for a node"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if direction == 'outgoing':
                cursor.execute(
                    "SELECT * FROM relationships WHERE source_node_id = ?",
                    (node_id,)
                )
            elif direction == 'incoming':
                cursor.execute(
                    "SELECT * FROM relationships WHERE target_node_id = ?",
                    (node_id,)
                )
            else:  # both
                cursor.execute(
                    "SELECT * FROM relationships WHERE source_node_id = ? OR target_node_id = ?",
                    (node_id, node_id)
                )
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_relationship_statistics(self) -> Dict:
        """Get statistics about relationships"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total relationships
            cursor.execute("SELECT COUNT(*) as total FROM relationships")
            total = cursor.fetchone()['total']
            
            # Relationships by type
            cursor.execute("""
                SELECT relationship_type, COUNT(*) as count
                FROM relationships
                GROUP BY relationship_type
                ORDER BY count DESC
            """)
            by_type = {row['relationship_type']: row['count'] for row in cursor.fetchall()}
            
            # Normalized types
            cursor.execute("""
                SELECT normalized_type, COUNT(*) as count
                FROM relationships
                GROUP BY normalized_type
                ORDER BY count DESC
            """)
            normalized = {row['normalized_type']: row['count'] for row in cursor.fetchall()}
            
            return {
                'total_relationships': total,
                'by_type': by_type,
                'by_normalized_type': normalized
            }
    
    # ==================== Node-Embedding Mapping ====================
    
    def link_node_to_embedding(
        self,
        node_id: str,
        embedding_id: str,
        relevance_score: Optional[float] = None
    ):
        """Link a graph node to an embedding"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO node_embeddings (
                    node_id, embedding_id, relevance_score
                ) VALUES (?, ?, ?)
            """, (node_id, embedding_id, relevance_score))
    
    def get_node_embeddings(self, node_id: str) -> List[Dict]:
        """Get all embeddings linked to a node"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT e.*, ne.relevance_score
                FROM embeddings e
                JOIN node_embeddings ne ON e.embedding_id = ne.embedding_id
                WHERE ne.node_id = ?
                ORDER BY ne.relevance_score DESC
            """, (node_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_embedding_nodes(self, embedding_id: str) -> List[Dict]:
        """Get all nodes linked to an embedding"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT gn.*, ne.relevance_score
                FROM graph_nodes gn
                JOIN node_embeddings ne ON gn.node_id = ne.node_id
                WHERE ne.embedding_id = ?
                ORDER BY ne.relevance_score DESC
            """, (embedding_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== Utility Operations ====================
    
    def get_database_statistics(self) -> Dict:
        """Get overall database statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Document count
            cursor.execute("SELECT COUNT(*) as count FROM documents")
            stats['documents'] = cursor.fetchone()['count']
            
            # Embedding count
            cursor.execute("SELECT COUNT(*) as count FROM embeddings")
            stats['embeddings'] = cursor.fetchone()['count']
            
            # Node count
            cursor.execute("SELECT COUNT(*) as count FROM graph_nodes")
            stats['graph_nodes'] = cursor.fetchone()['count']
            
            # Relationship count
            cursor.execute("SELECT COUNT(*) as count FROM relationships")
            stats['relationships'] = cursor.fetchone()['count']
            
            # Database size
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            stats['database_size_bytes'] = cursor.fetchone()['size']
            
            return stats
    
    def vacuum(self):
        """Optimize database by rebuilding it"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("VACUUM")
            log.info("Database vacuumed")


# Global instance
_metadata_db = None

def get_metadata_db() -> MetadataDatabase:
    """Get or create global metadata database instance"""
    global _metadata_db
    if _metadata_db is None:
        _metadata_db = MetadataDatabase()
    return _metadata_db
