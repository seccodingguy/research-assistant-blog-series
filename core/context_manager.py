# core/context_manager.py
from llama_index.core import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from typing import List, Dict, Optional
from datetime import datetime

from config import settings
from utils.logger import log


class ContextManager:
    """Manage context retrieval and enhancement for queries"""
    
    def __init__(self, index, memory_manager, graph_manager=None):
        self.index = index
        self.memory_manager = memory_manager
        self.graph_manager = graph_manager
        
        # Custom prompts
        self.qa_prompt_template = PromptTemplate(
            """You are an AI assistant with access to a knowledge base of PDF documents.
            
Context from documents:
{context_str}

Conversation history:
{conversation_history}

User question: {query_str}

Instructions:
1. Answer based on the provided context and conversation history
2. ALWAYS cite specific documents by their file names when making claims
3. If analyzing multiple papers, discuss each one explicitly
4. Synthesize findings across documents when relevant
5. If the context doesn't contain relevant information, say so
6. Be comprehensive and detailed in your analysis

Answer:"""
        )
        
        # Initialize retriever
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=settings.SIMILARITY_TOP_K
        )
        
        # Post-processor to filter by similarity
        self.postprocessor = SimilarityPostprocessor(
            similarity_cutoff=0.5
        )
        
        # Query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            node_postprocessors=[self.postprocessor]
        )
        
        log.info("ContextManager initialized")
    
    def retrieve_context(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        use_graph: bool = True
    ) -> List[Dict]:
        """Retrieve relevant context for a query with hybrid vector+graph"""
        try:
            if top_k is None:
                top_k = settings.SIMILARITY_TOP_K
            
            # Update retriever top_k
            self.retriever.similarity_top_k = top_k
            
            # Retrieve nodes from vector store
            nodes = self.retriever.retrieve(query)
            
            # Format results
            contexts = []
            for node in nodes:
                contexts.append({
                    "text": node.node.text,
                    "score": node.score,
                    "metadata": node.node.metadata,
                    "node_id": node.node.node_id
                })
            
            log.info(f"Retrieved {len(contexts)} vector contexts")
            
            # Add graph context if available
            graph_contexts = []
            if use_graph and self.graph_manager is not None:
                try:
                    graph_result = self.graph_manager.query_graph(
                        query,
                        similarity_top_k=5
                    )
                    
                    # Add graph source nodes as additional context
                    for source_node in graph_result.get('source_nodes', []):
                        graph_contexts.append({
                            "text": source_node['text'],
                            "score": source_node.get('score', 0.5),
                            "metadata": source_node.get('metadata', {}),
                            "node_id": f"graph_{len(graph_contexts)}",
                            "source": "knowledge_graph"
                        })
                    
                    if graph_contexts:
                        log.info(f"Retrieved {len(graph_contexts)} graph contexts")
                        contexts.extend(graph_contexts)
                        
                except Exception as e:
                    log.warning(f"Graph retrieval failed: {e}")
            
            return contexts
            
        except Exception as e:
            log.error(f"Error retrieving context: {e}")
            return []
    
    def build_enhanced_context(
        self, 
        query: str, 
        include_memory: bool = True,
        top_k: Optional[int] = None
    ) -> Dict:
        """Build enhanced context including documents and conversation history"""
        
        # Get document context with optional custom top_k
        doc_contexts = self.retrieve_context(query, top_k=top_k)
        
        # Get conversation context
        conversation_context = ""
        if include_memory:
            conversation_context = self.memory_manager.get_conversation_context()
        
        # Format document context
        doc_context_text = "\n\n".join([
            f"[Document: {ctx['metadata'].get('file_name', 'Unknown')}]\n{ctx['text']}"
            for ctx in doc_contexts
        ])
        
        # Extract unique sources
        sources = []
        seen_files = set()
        for ctx in doc_contexts:
            file_name = ctx['metadata'].get('file_name', 'Unknown')
            if file_name not in seen_files and file_name != 'Unknown':
                sources.append({
                    "file_name": file_name,
                    "file_path": ctx['metadata'].get('file_path', ''),
                    "relevance_score": ctx['score']
                })
                seen_files.add(file_name)
        
        enhanced_context = {
            "query": query,
            "document_context": doc_context_text,
            "conversation_context": conversation_context,
            "retrieved_chunks": len(doc_contexts),
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }
        
        return enhanced_context
    
    def format_prompt(
        self, 
        query: str, 
        context: Dict
    ) -> str:
        """Format the final prompt with all context"""
        
        prompt = self.qa_prompt_template.format(
            query_str=query,
            context_str=context["document_context"],
            conversation_history=context["conversation_context"]
        )
        
        return prompt
    
    def get_context_sources(
        self, 
        query: str
    ) -> List[Dict]:
        """Get source documents for a query"""
        contexts = self.retrieve_context(query)
        
        sources = []
        seen_files = set()
        
        for ctx in contexts:
            file_name = ctx['metadata'].get('file_name', 'Unknown')
            if file_name not in seen_files:
                sources.append({
                    "file_name": file_name,
                    "file_path": ctx['metadata'].get('file_path', ''),
                    "relevance_score": ctx['score']
                })
                seen_files.add(file_name)
        
        return sources
    
    def retrieve_all_documents(self, chunks_per_doc: int = 3) -> List[Dict]:
        """Retrieve representative chunks from ALL documents in the index.
        
        Args:
            chunks_per_doc: Number of chunks to retrieve per document
            
        Returns:
            List of context dictionaries with chunks from all documents
        """
        import chromadb
        from config import settings
        
        try:
            # Connect to ChromaDB directly
            chroma_client = chromadb.PersistentClient(path=str(settings.VECTOR_STORE_PATH))
            collection = chroma_client.get_collection(name=settings.COLLECTION_NAME)
            
            # Get all documents
            all_results = collection.get(include=['metadatas', 'documents', 'embeddings'])
            
            # Group chunks by document
            docs_map = {}
            for i, meta in enumerate(all_results['metadatas']):
                if meta and 'file_name' in meta:
                    file_name = meta['file_name']
                    if file_name != 'Unknown':
                        if file_name not in docs_map:
                            docs_map[file_name] = []
                        docs_map[file_name].append({
                            'id': all_results['ids'][i],
                            'text': all_results['documents'][i],
                            'metadata': meta,
                            'index': i
                        })
            
            log.info(f"Found {len(docs_map)} unique documents")
            
            # Get representative chunks from each document
            all_contexts = []
            for file_name, chunks in sorted(docs_map.items()):
                # Take first N chunks from each document (usually includes title/abstract)
                selected_chunks = chunks[:chunks_per_doc]
                
                for chunk in selected_chunks:
                    all_contexts.append({
                        'text': chunk['text'],
                        'score': 1.0,  # No similarity scoring when retrieving all
                        'metadata': chunk['metadata'],
                        'node_id': chunk['id']
                    })
            
            log.info(f"Retrieved {len(all_contexts)} chunks from {len(docs_map)} documents")
            return all_contexts
            
        except Exception as e:
            log.error(f"Error retrieving all documents: {e}")
            return []
    
    def summarize_context(
        self, 
        contexts: List[Dict]
    ) -> str:
        """Create a summary of retrieved contexts"""
        
        if not contexts:
            return "No relevant context found."
        
        summary_parts = [
            f"Found {len(contexts)} relevant passages from {len(set(c['metadata'].get('file_name') for c in contexts))} documents."
        ]
        
        # Group by document
        docs = {}
        for ctx in contexts:
            file_name = ctx['metadata'].get('file_name', 'Unknown')
            if file_name not in docs:
                docs[file_name] = []
            docs[file_name].append(ctx)
        
        for doc_name, doc_contexts in docs.items():
            avg_score = sum(c['score'] for c in doc_contexts) / len(doc_contexts)
            summary_parts.append(
                f"- {doc_name}: {len(doc_contexts)} passages (avg relevance: {avg_score:.2f})"
            )
        
        return "\n".join(summary_parts)
    
    def get_knowledge_graph_data(self, query: Optional[str] = None) -> Dict:
        """Get knowledge graph data for visualization"""
        if self.graph_manager is None:
            return {"nodes": [], "edges": [], "error": "Graph not initialized"}
        
        if query:
            # Get subgraph relevant to query
            result = self.graph_manager.query_graph(query)
            return {
                "nodes": result.get('nodes', []),
                "edges": result.get('relationships', []),
                "query": query
            }
        else:
            # Get full graph visualization
            return self.graph_manager.visualize_graph(max_nodes=50)