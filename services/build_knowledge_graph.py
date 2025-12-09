#!/usr/bin/env python3
"""
Script to build knowledge graph from existing documents in the vector store
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_parser import PDFParser
from core.graph_manager import GraphManager
from config import settings
from utils.logger import log
import chromadb
from llama_index.core import Document, Settings
import os
import logging 

logfilename = os.path.join("logs", "build_knowledge_graph.log")

# Configure logging
logging.basicConfig(
    filename=logfilename,
    level=logging.INFO,  # Changed from DEBUG to INFO to prevent log spam
    filemode="w",
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
FileOutputHandler = logging.FileHandler(logfilename)


def initialize_llm():
    """Initialize LLM and Embedding settings from config"""
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
                log.info(f"✓ Poe LLM configured: {settings.POE_MODEL_NAME}")
            else:
                log.error("Poe API key not found!")
                return False
                
        elif settings.LLM_PROVIDER.lower() == "ollama":
            from core.ollama_wrapper import OllamaLLM
            
            Settings.llm = OllamaLLM(
                base_url=settings.OLLAMA_BASE_URL,
                model_name=settings.OLLAMA_CHAT_MODEL,
                max_tokens=settings.OLLAMA_MAX_TOKENS,
                temperature=settings.OLLAMA_TEMPERATURE
            )
            log.info(f"✓ Ollama LLM configured: {settings.OLLAMA_CHAT_MODEL}")
        else:
            log.error(f"Unknown LLM provider: {settings.LLM_PROVIDER}")
            return False
        
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
            logger.info("✓ Azure OpenAI Embeddings configured")
            
        elif settings.EMBEDDING_PROVIDER.lower() == "ollama":
            from core.ollama_wrapper import OllamaEmbedding
            
            Settings.embed_model = OllamaEmbedding(
                base_url=settings.OLLAMA_BASE_URL,
                model_name=settings.OLLAMA_EMBEDDING_MODEL
            )
            logger.info(f"✓ Ollama Embeddings configured: {settings.OLLAMA_EMBEDDING_MODEL}")
        else:
            logger.error(f"Unknown embedding provider: {settings.EMBEDDING_PROVIDER}")
            return False
        
        return True
        
    except Exception as e:
        log.error(f"Failed to initialize LLM: {e}")
        return False


def build_graph_from_existing_documents():
    """Build knowledge graph from all documents in vector store"""

    logger.info("=" * 60)
    logger.info("Building Knowledge Graph from Existing Documents")
    logger.info("=" * 60)

    # Initialize LLM first
    logger.info("Initializing LLM...")
    if not initialize_llm():
        logger.error("Failed to initialize LLM. Cannot build knowledge graph.")
        return False
    
    try:
        # Initialize graph manager
        logger.info("Initializing GraphManager...")
        graph_manager = GraphManager()
        
        # Connect to ChromaDB
        logger.info("Connecting to ChromaDB...")
        chroma_client = chromadb.PersistentClient(
            path=str(settings.VECTOR_STORE_PATH)
        )
        
        # Get collection
        try:
            collection = chroma_client.get_collection(
                name=settings.COLLECTION_NAME
            )
            doc_count = collection.count()
            logger.info(f"Found collection '{settings.COLLECTION_NAME}' with {doc_count} documents")
        except Exception as e:
            logger.error(f"Could not get collection: {e}")
            return False
        
        if doc_count == 0:
            logger.warning("No documents in collection")
            return False
        
        # Get all documents
        logger.info("Retrieving all documents from vector store...")
        all_data = collection.get(
            include=['metadatas', 'documents']
        )
        
        # Group by file to avoid duplicates
        logger.info("Grouping documents by file...")
        docs_by_file = {}
        for i, (doc_id, text, metadata) in enumerate(zip(
            all_data['ids'],
            all_data['documents'],
            all_data['metadatas']
        )):
            if metadata and 'file_name' in metadata:
                file_name = metadata['file_name']
                if file_name not in docs_by_file:
                    docs_by_file[file_name] = []
                
                docs_by_file[file_name].append({
                    'id': doc_id,
                    'text': text,
                    'metadata': metadata
                })

        logger.info(f"Found {len(docs_by_file)} unique files")

        # Create Document objects for each file
        # Combine chunks from same file into single document
        # Use fewer chunks for faster processing
        logger.info("Creating Document objects...")
        documents = []
        for file_name, chunks in docs_by_file.items():
            # Combine first 2 chunks to get good representation
            # This reduces processing time significantly
            combined_text = "\n\n".join([
                chunk['text'] for chunk in chunks[:2]
            ])
            
            doc = Document(
                text=combined_text,
                metadata=chunks[0]['metadata'],
                id_=f"graph_{file_name}"
            )
            documents.append(doc)

        logger.info(f"Created {len(documents)} document objects")

        # Build knowledge graph with lower triplet count for faster processing
        logger.info("Building knowledge graph (this may take a while)...")
        logger.info("The LLM will extract entities and relationships...")
        logger.info(f"Estimated time: {len(documents) * 2-5} minutes")
        
        graph_manager.build_graph_from_documents(
            documents,
            max_triplets_per_chunk=3  # Reduced from 10 for faster processing
        )
        
        # Get statistics
        stats = graph_manager.get_graph_statistics()

        logger.info("=" * 60)
        logger.info("Knowledge Graph Build Complete!")
        logger.info("=" * 60)
        logger.info(f"Total Nodes: {stats.get('total_nodes', 0)}")
        logger.info(f"Total Edges: {stats.get('total_edges', 0)}")
        logger.info(f"Average Degree: {stats.get('avg_degree', 0):.2f}")
        logger.info(f"Graph Density: {stats.get('density', 0):.4f}")
        
        if stats.get('node_types'):
            logger.info("\nNode Types:")
            for node_type, count in stats['node_types'].items():
                logger.info(f"  {node_type}: {count}")
        
        # Save visualization
        output_path = Path("./outputs/initial_graph_viz.json")
        logger.info(f"\nSaving graph visualization to {output_path}...")
        graph_manager.visualize_graph(output_path=output_path)

        logger.info("=" * 60)
        logger.info("Success! Knowledge graph is ready to use.")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error building knowledge graph: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Knowledge Graph Builder")
    print("=" * 60)
    print(f"Vector Store: {settings.VECTOR_STORE_PATH}")
    print(f"Collection: {settings.COLLECTION_NAME}")
    print(f"Graph Store: {settings.GRAPH_STORE_PATH}")
    print("=" * 60 + "\n")
    
    success = build_graph_from_existing_documents()
    
    sys.exit(0 if success else 1)
