# core/graph_manager.py
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import (
    KnowledgeGraphIndex,
    StorageContext,
    Document,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from enum import Enum
import json
import networkx as nx
from datetime import datetime

from config import settings
from utils.logger import log


class RelationType(Enum):
    """Types of relationships between scientific concepts"""
    CAUSES = "causes"
    CORRELATES_WITH = "correlates_with"
    INHIBITS = "inhibits"
    ENHANCES = "enhances"
    SUPPORTS = "supports"
    INTERACTS_WITH = "interacts_with"
    ASSOCIATED_WITH = "associated_with"


class ConceptType(Enum):
    """Types of scientific concepts"""
    AI = "artificial intelligence"
    DEEP_LEARNING = "deep learning"
    MACHINE_LEARNING = "machine learning"
    MODELS = "models"
    DATASET = "dataset"
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    NEURAL_NETWORK = "neural network"
    LLM = "large language model"
    PROMPT = "prompt"
    GENERATIVE_AI = "generative AI"
    EMBEDDING = "embedding"
    NLP = "natural language processing"
    COMPUTER_VISION = "computer vision"
    ROBOTICS = "robotics"
    CONTEXT = "context"
    SELF_LEARNING = "self-learning"
    MEMORY = "memory"
    AGENT = "agent"
    REASONING = "reasoning"
    ACCURACY = "accuracy"
    KNOWLEDGE = "knowledge"
    KNOWLEDGE_GRAPH = "knowledge graph"
    HALLUCINATION = "hallucination"
    GENERAL = "general"
    PROCESS = "process"
    METRIC = "metric"
    OTHER = "other"


# Simple keyword extraction for common research concepts
CONCEPT_KEYWORDS = {
    'knowledge graph': ConceptType.KNOWLEDGE_GRAPH,
    'knowledge graphs': ConceptType.KNOWLEDGE_GRAPH,
    'memory': ConceptType.MEMORY,
    'agent': ConceptType.AGENT,
    'agents': ConceptType.AGENT,
    'agentic': ConceptType.AGENT,
    'accuracy': ConceptType.ACCURACY,
    'neural network': ConceptType.NEURAL_NETWORK,
    'machine learning': ConceptType.MACHINE_LEARNING,
    'deep learning': ConceptType.DEEP_LEARNING,
    'transformer': ConceptType.MODELS,
    'transformers': ConceptType.MODELS,
    'embedding': ConceptType.EMBEDDING,
    'embeddings': ConceptType.EMBEDDING,
    'reasoning': ConceptType.REASONING,
    'llm': ConceptType.LLM,
    'large language model': ConceptType.LLM,
    'generative ai': ConceptType.GENERATIVE_AI,
    'artificial intelligence': ConceptType.AI,
    'nlp': ConceptType.NLP,
    'natural language processing': ConceptType.NLP,
    'computer vision': ConceptType.COMPUTER_VISION,
    'robotics': ConceptType.ROBOTICS,
    'supervised learning': ConceptType.SUPERVISED,
    'unsupervised learning': ConceptType.UNSUPERVISED,
    'reinforcement learning': ConceptType.REINFORCEMENT,
    'dataset': ConceptType.DATASET,
    'datasets': ConceptType.DATASET,
    'hallucination': ConceptType.HALLUCINATION,
    'prompt': ConceptType.PROMPT,
    'prompting': ConceptType.PROMPT,
}


class GraphManager:
    """Manage knowledge graph for enhanced context retrieval"""
    
    def __init__(self):
        log.info("Initializing GraphManager...")
        
        # Initialize graph store
        self.graph_store_path = settings.GRAPH_STORE_PATH / "graph_store.json"
        self.graph_store = SimpleGraphStore()
        
        # Load existing graph if available
        if self.graph_store_path.exists():
            try:
                self.graph_store = SimpleGraphStore.from_persist_path(
                    str(settings.GRAPH_STORE_PATH)
                )
                log.info("✓ Loaded existing graph store")
            except Exception as e:
                log.warning(f"Could not load graph store: {e}")
                self.graph_store = SimpleGraphStore()
        
        # Create storage context with graph store
        self.storage_context = StorageContext.from_defaults(
            graph_store=self.graph_store
        )
        
        # Initialize knowledge graph index (will be set during build)
        self.kg_index = None
        
        # NetworkX graph for analysis and visualization
        self.nx_graph = nx.DiGraph()
        
        log.info("✓ GraphManager initialized")
    
    def build_graph_from_documents(
        self, 
        documents: List[Document],
        max_triplets_per_chunk: int = 10
    ) -> None:
        """Build knowledge graph from documents using LLM extraction"""
        try:
            log.info(f"Building knowledge graph from {len(documents)} documents...")
            
            # Create knowledge graph index with LLM-based extraction
            self.kg_index = KnowledgeGraphIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                max_triplets_per_chunk=max_triplets_per_chunk,
                include_embeddings=True,
                show_progress=True
            )
            
            # Persist the graph
            self.save_graph()
            
            # Update NetworkX graph for analysis
            self._sync_to_networkx()
            
            log.info(f"✓ Knowledge graph built with {len(self.nx_graph.nodes)} nodes and {len(self.nx_graph.edges)} edges")
            
        except Exception as e:
            log.error(f"Error building knowledge graph: {e}")
            raise
    
    def add_documents_to_graph(
        self, 
        documents: List[Document],
        max_triplets_per_chunk: int = 10
    ) -> None:
        """Add new documents to existing knowledge graph"""
        try:
            log.info(f"Adding {len(documents)} documents to knowledge graph...")
            
            if self.kg_index is None:
                # No existing index, build new one
                self.build_graph_from_documents(documents, max_triplets_per_chunk)
            else:
                # Add to existing index
                for doc in documents:
                    self.kg_index.insert(doc)
                
                # Persist the updated graph
                self.save_graph()
                
                # Update NetworkX graph
                self._sync_to_networkx()
                
                log.info(f"✓ Added documents to graph. Now has {len(self.nx_graph.nodes)} nodes and {len(self.nx_graph.edges)} edges")
                
        except Exception as e:
            log.error(f"Error adding documents to graph: {e}")
            raise
    
    def extract_concepts_from_text(self, text: str) -> List[Tuple[str, ConceptType]]:
        """Extract research concepts from text using keyword matching"""
        concepts = []
        text_lower = text.lower()
        
        for keyword, concept_type in CONCEPT_KEYWORDS.items():
            if keyword in text_lower:
                concepts.append((keyword, concept_type))
        
        return concepts
    
    def query_graph(
        self, 
        query: str,
        similarity_top_k: int = 5,
        include_text: bool = True
    ) -> Dict:
        """Query the knowledge graph"""
        try:
            if self.kg_index is None:
                log.warning("Knowledge graph not initialized")
                return {
                    "nodes": [],
                    "relationships": [],
                    "context": ""
                }
            
            # Query the knowledge graph index
            query_engine = self.kg_index.as_query_engine(
                include_text=include_text,
                similarity_top_k=similarity_top_k,
                response_mode="tree_summarize"
            )
            
            response = query_engine.query(query)
            
            # Extract relevant subgraph
            subgraph_nodes, subgraph_edges = self._extract_relevant_subgraph(query)
            
            return {
                "response": str(response),
                "nodes": subgraph_nodes,
                "relationships": subgraph_edges,
                "source_nodes": [
                    {
                        "text": node.text,
                        "score": node.score,
                        "metadata": node.metadata
                    }
                    for node in response.source_nodes
                ] if hasattr(response, 'source_nodes') else []
            }
            
        except Exception as e:
            log.error(f"Error querying graph: {e}")
            return {
                "nodes": [],
                "relationships": [],
                "context": "",
                "error": str(e)
            }
    
    def get_related_concepts(
        self, 
        concept: str,
        max_depth: int = 2
    ) -> List[Dict]:
        """Get concepts related to a given concept"""
        try:
            if concept not in self.nx_graph:
                return []
            
            # Get neighbors within max_depth
            related = []
            visited = set()
            queue = [(concept, 0)]
            
            while queue:
                current, depth = queue.pop(0)
                
                if current in visited or depth > max_depth:
                    continue
                
                visited.add(current)
                
                if depth > 0:  # Don't include the original concept
                    related.append({
                        "concept": current,
                        "depth": depth,
                        "type": self.nx_graph.nodes[current].get('type', 'unknown')
                    })
                
                # Add neighbors
                for neighbor in self.nx_graph.neighbors(current):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
            
            return related
            
        except Exception as e:
            log.error(f"Error getting related concepts: {e}")
            return []
    
    def get_graph_statistics(self) -> Dict:
        """Get statistics about the knowledge graph"""
        try:
            stats = {
                "total_nodes": len(self.nx_graph.nodes),
                "total_edges": len(self.nx_graph.edges),
                "avg_degree": sum(dict(self.nx_graph.degree()).values()) / len(self.nx_graph.nodes) if self.nx_graph.nodes else 0,
                "density": nx.density(self.nx_graph),
                "is_connected": nx.is_weakly_connected(self.nx_graph) if self.nx_graph.nodes else False
            }
            
            # Get node types distribution
            node_types = {}
            for node, data in self.nx_graph.nodes(data=True):
                node_type = data.get('type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            stats['node_types'] = node_types
            
            return stats
            
        except Exception as e:
            log.error(f"Error getting graph statistics: {e}")
            return {}
    
    def visualize_graph(
        self, 
        output_path: Optional[Path] = None,
        max_nodes: int = 100
    ) -> Dict:
        """Generate graph visualization data"""
        try:
            # Limit nodes for visualization
            if len(self.nx_graph.nodes) > max_nodes:
                # Get most connected nodes
                node_degrees = dict(self.nx_graph.degree())
                top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
                subgraph = self.nx_graph.subgraph([n[0] for n in top_nodes])
            else:
                subgraph = self.nx_graph
            
            # Prepare data for visualization
            nodes = []
            for node, data in subgraph.nodes(data=True):
                nodes.append({
                    "id": node,
                    "label": node,
                    "type": data.get('type', 'unknown'),
                    "degree": subgraph.degree(node)
                })
            
            edges = []
            for source, target, data in subgraph.edges(data=True):
                edges.append({
                    "source": source,
                    "target": target,
                    "label": data.get('relationship', 'related_to')
                })
            
            viz_data = {
                "nodes": nodes,
                "edges": edges,
                "stats": {
                    "node_count": len(nodes),
                    "edge_count": len(edges)
                }
            }
            
            # Save to file if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(viz_data, f, indent=2)
                log.info(f"✓ Graph visualization data saved to {output_path}")
            
            return viz_data
            
        except Exception as e:
            log.error(f"Error visualizing graph: {e}")
            return {"nodes": [], "edges": [], "error": str(e)}
    
    def save_graph(self) -> None:
        """Persist the knowledge graph to disk"""
        try:
            settings.GRAPH_STORE_PATH.mkdir(parents=True, exist_ok=True)
            self.graph_store.persist(
                persist_path=str(settings.GRAPH_STORE_PATH)
            )
            log.info("✓ Knowledge graph saved to disk")
        except Exception as e:
            log.error(f"Error saving graph: {e}")
    
    def load_graph(self) -> bool:
        """Load knowledge graph from disk"""
        try:
            if not self.graph_store_path.exists():
                log.info("No existing graph store found")
                return False
            
            self.graph_store = SimpleGraphStore.from_persist_path(
                str(settings.GRAPH_STORE_PATH)
            )
            
            # Recreate storage context
            self.storage_context = StorageContext.from_defaults(
                graph_store=self.graph_store
            )
            
            # Sync to NetworkX
            self._sync_to_networkx()
            
            log.info("✓ Knowledge graph loaded from disk")
            return True
            
        except Exception as e:
            log.error(f"Error loading graph: {e}")
            return False
    
    def _sync_to_networkx(self) -> None:
        """Sync SimpleGraphStore to NetworkX graph for analysis"""
        try:
            self.nx_graph.clear()
            
            # Get all triplets from graph store
            if hasattr(self.graph_store, 'get_all_triplets'):
                triplets = self.graph_store.get_all_triplets()
                for subj, rel, obj in triplets:
                    self.nx_graph.add_edge(
                        subj,
                        obj,
                        relationship=rel
                    )
            
            log.info(f"✓ Synced to NetworkX: {len(self.nx_graph.nodes)} nodes, {len(self.nx_graph.edges)} edges")
            
        except Exception as e:
            log.error(f"Error syncing to NetworkX: {e}")
    
    def _extract_relevant_subgraph(
        self, 
        query: str,
        max_nodes: int = 20
    ) -> Tuple[List[Dict], List[Dict]]:
        """Extract relevant subgraph based on query"""
        try:
            # Extract concepts from query
            query_concepts = self.extract_concepts_from_text(query)
            
            if not query_concepts:
                return [], []
            
            # Find nodes matching query concepts
            relevant_nodes = set()
            for concept_keyword, concept_type in query_concepts:
                for node in self.nx_graph.nodes():
                    if concept_keyword.lower() in node.lower():
                        relevant_nodes.add(node)
            
            # If we have relevant nodes, get their neighborhoods
            if relevant_nodes:
                expanded_nodes = set(relevant_nodes)
                for node in list(relevant_nodes)[:max_nodes]:
                    # Add immediate neighbors
                    expanded_nodes.update(self.nx_graph.neighbors(node))
                
                # Limit size
                expanded_nodes = list(expanded_nodes)[:max_nodes]
                
                # Extract subgraph
                subgraph = self.nx_graph.subgraph(expanded_nodes)
                
                # Format nodes
                nodes = [
                    {
                        "id": node,
                        "label": node,
                        "type": subgraph.nodes[node].get('type', 'unknown')
                    }
                    for node in subgraph.nodes()
                ]
                
                # Format edges
                edges = [
                    {
                        "source": source,
                        "target": target,
                        "relationship": data.get('relationship', 'related_to')
                    }
                    for source, target, data in subgraph.edges(data=True)
                ]
                
                return nodes, edges
            
            return [], []
            
        except Exception as e:
            log.error(f"Error extracting subgraph: {e}")
            return [], []
