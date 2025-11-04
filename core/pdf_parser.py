# core/pdf_parser.py
from llama_index.core import (
    SimpleDirectoryReader,
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    KeywordExtractor
)
from llama_index.core.ingestion import IngestionPipeline
from core.azure_openai_wrapper import AzureOpenAIEmbedding
from core.graph_manager import GraphManager
from core.metadata_db import get_metadata_db

import chromadb
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
from datetime import datetime
import uuid

from config import settings
from utils.logger import log


class PDFParser:
    """Advanced PDF Parser with metadata extraction and chunking"""
    
    def __init__(self):
        log.info("Initializing PDFParser...")
        
        # Configure global Settings based on provider selection
        try:
            # Configure LLM based on provider selection
            if settings.LLM_PROVIDER.lower() == "poe":
                from core.azure_openai_wrapper import PoeLLM
                
                if settings.POE_API_KEY:
                    Settings.llm = PoeLLM(
                        api_key=settings.POE_API_KEY,
                        model_name=settings.POE_MODEL_NAME,
                        max_tokens=settings.POE_MAX_TOKENS,
                        temperature=settings.POE_TEMPERATURE
                    )
                    log.info(f"✓ Poe LLM configured with model: {settings.POE_MODEL_NAME}")
                else:
                    Settings.llm = None
                    log.warning("⚠ Poe API key not found, LLM disabled")
                    
            elif settings.LLM_PROVIDER.lower() == "ollama":
                from core.ollama_wrapper import OllamaLLM
                
                Settings.llm = OllamaLLM(
                    base_url=settings.OLLAMA_BASE_URL,
                    model_name=settings.OLLAMA_CHAT_MODEL,
                    max_tokens=settings.OLLAMA_MAX_TOKENS,
                    temperature=settings.OLLAMA_TEMPERATURE
                )
                log.info(f"✓ Ollama LLM configured with model: {settings.OLLAMA_CHAT_MODEL}")
            else:
                Settings.llm = None
                log.warning(f"⚠ Unknown LLM provider: {settings.LLM_PROVIDER}")
            
            # Configure embeddings based on provider selection
            if settings.EMBEDDING_PROVIDER.lower() == "azure":
                from core.azure_openai_wrapper import AzureOpenAIEmbedding
                
                Settings.embed_model = AzureOpenAIEmbedding(
                    api_key=settings.AZURE_OPENAI_API_KEY,
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                    deployment_name=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                    dimensions=settings.AZURE_OPENAI_EMBEDDING_DIMENSION
                )
                log.info("✓ Azure OpenAI Embeddings configured successfully")
                
            elif settings.EMBEDDING_PROVIDER.lower() == "ollama":
                from core.ollama_wrapper import OllamaEmbedding
                
                Settings.embed_model = OllamaEmbedding(
                    base_url=settings.OLLAMA_BASE_URL,
                    model_name=settings.OLLAMA_EMBEDDING_MODEL
                )
                log.info(f"✓ Ollama Embeddings configured with model: {settings.OLLAMA_EMBEDDING_MODEL}")
            else:
                log.error(f"Unknown embedding provider: {settings.EMBEDDING_PROVIDER}")
                raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")
                
        except Exception as e:
            log.error(f"Failed to configure AI providers: {e}")
            raise
        
        Settings.chunk_size = settings.CHUNK_SIZE
        Settings.chunk_overlap = settings.CHUNK_OVERLAP
        log.info(f"✓ Chunk settings: size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP}")
        
        # Initialize graph manager for knowledge graph
        try:
            self.graph_manager = GraphManager()
            log.info("✓ GraphManager initialized")
        except Exception as e:
            log.warning(f"GraphManager initialization failed: {e}")
            self.graph_manager = None
        
        # Initialize metadata database
        try:
            self.metadata_db = get_metadata_db()
            log.info("✓ Metadata database initialized")
        except Exception as e:
            log.warning(f"Metadata database initialization failed: {e}")
            self.metadata_db = None
        
        # Get expected embedding dimension based on provider
        if settings.EMBEDDING_PROVIDER.lower() == "azure":
            expected_dimension = settings.AZURE_OPENAI_EMBEDDING_DIMENSION
        elif settings.EMBEDDING_PROVIDER.lower() == "ollama":
            expected_dimension = 768  # nomic-embed-text dimension
        else:
            expected_dimension = None
        
        # Initialize Chroma client
        # Add settings to avoid readonly database issues in WSL/Windows mounts
        import chromadb.config
        chroma_settings = chromadb.config.Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True,
        )
        
        self.chroma_client = chromadb.PersistentClient(
            path=str(settings.VECTOR_STORE_PATH),
            settings=chroma_settings
        )
        
        # Check for dimension mismatch and handle collection migration
        self._validate_and_migrate_collection(settings.COLLECTION_NAME, expected_dimension)
        
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name=settings.COLLECTION_NAME
        )
        
        # Vector store
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.chroma_collection
        )
        
        # Storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # Node parser with metadata extractors
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        # Metadata extractors
        # Note: All metadata extractors removed to avoid LLM API calls during document processing
        # This prevents unnecessary Azure OpenAI API calls and potential 404 errors
        # If you need metadata extraction, ensure your Azure deployment has a valid GPT-4 endpoint
        self.extractors = []
        
        # Ingestion pipeline
        self.pipeline = IngestionPipeline(
            transformations=[
                self.node_parser,
                *self.extractors,
                Settings.embed_model
            ]
        )
        
        self.index = None
        log.info("PDFParser initialized successfully")
    
    def _validate_and_migrate_collection(self, collection_name: str, expected_dimension: Optional[int]):
        """
        Validate embedding dimensions and migrate collection if mismatch detected.
        Preserves all documents by re-embedding them with the new provider.
        """
        try:
            # Try to get existing collection
            try:
                existing_collection = self.chroma_client.get_collection(collection_name)
                doc_count = existing_collection.count()
                
                if doc_count == 0:
                    log.info(f"Collection '{collection_name}' is empty, no migration needed")
                    return
                
                # Check embedding dimension
                result = existing_collection.peek(1)
                if result and 'embeddings' in result and len(result['embeddings']) > 0:
                    current_dimension = len(result['embeddings'][0])
                    
                    if expected_dimension and current_dimension != expected_dimension:
                        log.warning(f"Embedding dimension mismatch detected!")
                        log.warning(f"  Current: {current_dimension}, Expected: {expected_dimension}")
                        log.warning(f"  This will cause query failures.")
                        log.warning(f"  Collection '{collection_name}' needs to be rebuilt.")
                        log.warning(f"  Documents: {doc_count}")
                        
                        # For now, log the issue. Auto-rebuild can be triggered by user command
                        log.info("To rebuild with new embeddings, use: process <pdf_folder>")
                        log.info("Or manually clear the vector store and re-index documents")
                    else:
                        log.info(f"✓ Embedding dimension validated: {current_dimension}")
                        
            except ValueError:
                # Collection doesn't exist, will be created
                log.info(f"Collection '{collection_name}' will be created on first use")
                
        except Exception as e:
            log.error(f"Error validating collection: {e}")
    
    def rebuild_index_with_new_embeddings(self):
        """
        Rebuild the entire index with new embedding provider.
        Extracts all document texts and re-embeds them.
        """
        try:
            log.info("Starting index rebuild with new embeddings...")
            
            # Get existing collection
            try:
                old_collection = self.chroma_client.get_collection(settings.COLLECTION_NAME)
                doc_count = old_collection.count()
                
                if doc_count == 0:
                    log.info("No documents to rebuild")
                    return
                
                log.info(f"Found {doc_count} documents to re-embed")
                
                # Get all documents
                all_data = old_collection.get()
                
                # Extract texts and metadata
                documents = []
                for i, (doc_id, text, metadata) in enumerate(zip(
                    all_data['ids'],
                    all_data['documents'],
                    all_data['metadatas']
                )):
                    if text:
                        doc = Document(
                            text=text,
                            metadata=metadata or {},
                            id_=doc_id
                        )
                        documents.append(doc)
                
                log.info(f"Extracted {len(documents)} documents")
                
                # Delete old collection
                self.chroma_client.delete_collection(settings.COLLECTION_NAME)
                log.info(f"Deleted old collection")
                
                # Create new collection
                self.chroma_collection = self.chroma_client.get_or_create_collection(
                    name=settings.COLLECTION_NAME
                )
                
                # Update vector store
                self.vector_store = ChromaVectorStore(
                    chroma_collection=self.chroma_collection
                )
                
                # Update storage context
                self.storage_context = StorageContext.from_defaults(
                    vector_store=self.vector_store
                )
                
                # Process documents with new embeddings
                log.info("Re-embedding documents with new provider...")
                self.index = self.process_documents(documents)
                
                log.info(f"✓ Successfully rebuilt index with new embeddings")
                log.info(f"✓ Re-embedded {len(documents)} documents")
                
            except ValueError:
                log.info("Collection doesn't exist, nothing to rebuild")
                
        except Exception as e:
            log.error(f"Error rebuilding index: {e}")
            raise
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file for deduplication"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _is_file_processed(self, file_hash: str) -> bool:
        """Check if file has already been processed"""
        try:
            results = self.chroma_collection.get(
                where={"file_hash": file_hash},
                limit=1
            )
            return len(results['ids']) > 0
        except Exception as e:
            log.warning(f"Error checking file hash: {e}")
            return False
    
    def parse_pdf(self, file_path: Path) -> List[Document]:
        """Parse a single PDF file and extract documents"""
        try:
            log.info(f"Parsing PDF: {file_path.name}")
            
            # Check if already processed
            file_hash = self._calculate_file_hash(file_path)
            if self._is_file_processed(file_hash):
                log.info(f"File already processed: {file_path.name}")
                return []
            
            # Generate unique document ID
            document_id = str(uuid.uuid4())
            parsed_date = datetime.now().isoformat()
            
            # Load documents
            reader = SimpleDirectoryReader(
                input_files=[str(file_path)],
                file_metadata=lambda filename: {
                    "document_id": document_id,
                    "file_name": Path(filename).name,
                    "file_path": str(filename),
                    "file_hash": file_hash,
                    "parsed_date": parsed_date,
                    "file_type": "pdf"
                }
            )
            
            documents = reader.load_data()
            log.info(f"Loaded {len(documents)} documents from {file_path.name}")
            
            # Save document metadata to database
            if self.metadata_db is not None:
                try:
                    self.metadata_db.insert_document(
                        document_id=document_id,
                        file_name=file_path.name,
                        file_path=str(file_path.absolute()),
                        file_hash=file_hash,
                        parsed_date=parsed_date,
                        title=file_path.stem,  # Use filename as title for now
                        metadata={
                            "total_chunks": len(documents)
                        }
                    )
                    log.info(f"✓ Saved document metadata to database: {document_id}")
                except Exception as e:
                    log.warning(f"Failed to save document metadata: {e}")
            
            return documents
            
        except Exception as e:
            log.error(f"Error parsing PDF {file_path}: {e}")
            return []
    
    def parse_folder(self, folder_path: Optional[Path] = None) -> List[Document]:
        """Parse all PDFs in a folder"""
        if folder_path is None:
            folder_path = settings.DOWNLOADS_FOLDER
        
        log.info(f"Scanning folder: {folder_path}")
        
        all_documents = []
        pdf_files = list(folder_path.glob("*.pdf"))
        
        log.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            documents = self.parse_pdf(pdf_file)
            all_documents.extend(documents)
        
        log.info(f"Total documents extracted: {len(all_documents)}")
        return all_documents
    
    def process_documents(self, documents: List[Document]) -> VectorStoreIndex:
        """Process documents through the ingestion pipeline and create index"""
        try:
            if not documents:
                log.warning("No documents to process")
                return self.index
            
            log.info(f"Processing {len(documents)} documents through pipeline")
            
            # Run pipeline
            nodes = self.pipeline.run(documents=documents)
            log.info(f"Generated {len(nodes)} nodes from documents")
            
            # Save embedding metadata to database
            if self.metadata_db is not None:
                try:
                    for idx, node in enumerate(nodes):
                        # Extract document metadata
                        doc_metadata = node.metadata if hasattr(node, 'metadata') else {}
                        document_id = doc_metadata.get('document_id', str(uuid.uuid4()))
                        
                        # Generate unique embedding ID
                        embedding_id = node.node_id if hasattr(node, 'node_id') else str(uuid.uuid4())
                        
                        # Determine embedding provider and model
                        embedding_provider = settings.EMBEDDING_PROVIDER
                        if embedding_provider.lower() == "azure":
                            embedding_model = settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
                            vector_dimension = settings.AZURE_OPENAI_EMBEDDING_DIMENSION
                        elif embedding_provider.lower() == "ollama":
                            embedding_model = settings.OLLAMA_EMBEDDING_MODEL
                            vector_dimension = 768  # nomic-embed-text dimension
                        else:
                            embedding_model = "unknown"
                            vector_dimension = None
                        
                        # Insert embedding metadata
                        self.metadata_db.insert_embedding(
                            embedding_id=embedding_id,
                            document_id=document_id,
                            chunk_index=idx,
                            chunk_text=node.text if hasattr(node, 'text') else "",
                            chunk_size=len(node.text) if hasattr(node, 'text') else 0,
                            vector_dimension=vector_dimension,
                            embedding_provider=embedding_provider,
                            embedding_model=embedding_model,
                            vector_store_collection=settings.COLLECTION_NAME,
                            metadata=doc_metadata
                        )
                    
                    log.info(f"✓ Saved {len(nodes)} embedding records to database")
                    
                    # Update document chunk count
                    if documents and hasattr(documents[0], 'metadata'):
                        doc_id = documents[0].metadata.get('document_id')
                        if doc_id:
                            self.metadata_db.update_document_chunk_count(doc_id, len(nodes))
                
                except Exception as e:
                    log.warning(f"Failed to save embedding metadata: {e}")
            
            # Add documents to knowledge graph
            if self.graph_manager is not None:
                try:
                    log.info("Adding documents to knowledge graph...")
                    self.graph_manager.add_documents_to_graph(
                        documents,
                        max_triplets_per_chunk=15  # Increased for richer graph
                    )
                except Exception as e:
                    log.warning(f"Failed to add to knowledge graph: {e}")
            
            # Create or update index
            if self.index is None:
                self.index = VectorStoreIndex(
                    nodes=nodes,
                    storage_context=self.storage_context
                )
                log.info("Created new vector index")
            else:
                # Add nodes to existing index
                self.index.insert_nodes(nodes)
                log.info("Updated existing vector index")
            
            # Persist index
            self.index.storage_context.persist(
                persist_dir=str(settings.VECTOR_STORE_PATH)
            )
            log.info("Index persisted to disk")
            
            return self.index
            
        except Exception as e:
            log.error(f"Error processing documents: {e}")
            raise
    
    def load_existing_index(self) -> Optional[VectorStoreIndex]:
        """Load existing index from disk"""
        try:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store
            )
            log.info("Loaded existing index from disk")
            return self.index
        except Exception as e:
            log.warning(f"Could not load existing index: {e}")
            return None
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the current index"""
        try:
            collection_count = self.chroma_collection.count()
            return {
                "total_documents": collection_count,
                "collection_name": settings.COLLECTION_NAME,
                "vector_store_path": str(settings.VECTOR_STORE_PATH)
            }
        except Exception as e:
            log.error(f"Error getting index stats: {e}")
            return {}