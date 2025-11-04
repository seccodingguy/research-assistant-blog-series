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
import re
import networkx as nx
from datetime import datetime
import hashlib

from config import settings
from config.ontology_loader import get_ontology_loader
from core.metadata_db import get_metadata_db
from utils.logger import log


# ============================================================================
# DEPRECATED: The following Enums and constants are kept for backward
# compatibility only. All classification logic now uses the OntologyLoader
# with configuration from config/graph_ontology.yaml.
#
# To modify node types, relationships, or keywords:
# 1. Edit config/graph_ontology.yaml
# 2. Run: python main.py ontology reload
# 3. Or restart the application
#
# See documentation/ONTOLOGY_DECOUPLING_STRATEGY.md for details.
# ============================================================================

class RelationType(Enum):
    """Types of relationships between scientific concepts"""
    CAUSES = "causes"
    CORRELATES_WITH = "correlates_with"
    INHIBITS = "inhibits"
    ENHANCES = "enhances"
    SUPPORTS = "supports"
    INTERACTS_WITH = "interacts_with"
    ASSOCIATED_WITH = "associated_with"
    # Extended relationship types
    USES = "uses"
    IMPLEMENTS = "implements"
    DEFINES = "defines"
    PROVIDES = "provides"
    REQUIRES = "requires"
    ENABLES = "enables"
    CREATES = "creates"
    BELONGS_TO = "belongs_to"
    PART_OF = "part_of"
    AUTHORED_BY = "authored_by"
    PUBLISHED_IN = "published_in"
    REFERENCED_BY = "referenced_by"



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
    # New categories for better classification
    PROTOCOL = "protocol"
    COMMUNICATION = "communication"
    SECURITY = "security"
    AUTHENTICATION = "authentication"
    API = "api"
    FRAMEWORK = "framework"
    SYSTEM = "system"
    ARCHITECTURE = "architecture"
    NETWORK = "network"
    DATA = "data"
    MESSAGE = "message"
    TOKEN = "token"
    BLOCKCHAIN = "blockchain"
    PAYMENT = "payment"
    TRANSACTION = "transaction"
    IDENTITY = "identity"
    ENCRYPTION = "encryption"
    VALIDATION = "validation"
    AUTHORIZATION = "authorization"
    # Phase 1+2: New categories for better classification
    PERSON = "person"
    ROLE = "role"
    ACTOR = "actor"
    DOCUMENT_TYPE = "document_type"
    METHOD = "method"
    FORMAL_METHOD = "formal_method"
    TOOL = "tool"
    RESEARCH = "research"
    CRYPTOGRAPHY = "cryptography"
    DISTRIBUTED_SYSTEM = "distributed_system"
    ORGANIZATION = "organization"
    COMPANY = "company"
    MEASUREMENT = "measurement"
    TEMPORAL = "temporal"
    ALGORITHM = "algorithm"
    COMPONENT = "component"
    OTHER = "other"


# Simple keyword extraction for common research concepts
CONCEPT_KEYWORDS = {
    # AI & ML Core
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
    'language model': ConceptType.LLM,
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
    'model': ConceptType.MODELS,
    # Protocols & Communication
    'protocol': ConceptType.PROTOCOL,
    'a2a': ConceptType.PROTOCOL,
    'mcp': ConceptType.PROTOCOL,
    'anp': ConceptType.PROTOCOL,
    'json-rpc': ConceptType.PROTOCOL,
    'http': ConceptType.PROTOCOL,
    'https': ConceptType.PROTOCOL,
    'api': ConceptType.API,
    'rest': ConceptType.API,
    'communication': ConceptType.COMMUNICATION,
    'message': ConceptType.MESSAGE,
    'messaging': ConceptType.MESSAGE,
    # Security
    'security': ConceptType.SECURITY,
    'secure': ConceptType.SECURITY,
    'authentication': ConceptType.AUTHENTICATION,
    'identity': ConceptType.IDENTITY,
    'encryption': ConceptType.ENCRYPTION,
    'validation': ConceptType.VALIDATION,
    'authorization': ConceptType.AUTHORIZATION,
    'oauth': ConceptType.AUTHORIZATION,
    'jwt': ConceptType.TOKEN,
    'token': ConceptType.TOKEN,
    # Systems & Architecture
    'framework': ConceptType.FRAMEWORK,
    'system': ConceptType.SYSTEM,
    'architecture': ConceptType.ARCHITECTURE,
    'network': ConceptType.NETWORK,
    'decentralized': ConceptType.NETWORK,
    'peer-to-peer': ConceptType.NETWORK,
    'client-server': ConceptType.ARCHITECTURE,
    # Blockchain & Payments
    'blockchain': ConceptType.BLOCKCHAIN,
    'payment': ConceptType.PAYMENT,
    'transaction': ConceptType.TRANSACTION,
    'micropayment': ConceptType.PAYMENT,
    # Data
    'data': ConceptType.DATA,
    'metadata': ConceptType.DATA,
    'context': ConceptType.CONTEXT,
    
    # Phase 1+2: Research & Documents
    'paper': ConceptType.DOCUMENT_TYPE,
    'document': ConceptType.DOCUMENT_TYPE,
    'survey': ConceptType.RESEARCH,
    'study': ConceptType.RESEARCH,
    'research': ConceptType.RESEARCH,
    'publication': ConceptType.DOCUMENT_TYPE,
    'article': ConceptType.DOCUMENT_TYPE,
    'thesis': ConceptType.DOCUMENT_TYPE,
    'report': ConceptType.DOCUMENT_TYPE,
    
    # Phase 1+2: Methods & Analysis
    'analysis': ConceptType.METHOD,
    'evaluation': ConceptType.METHOD,
    'experiment': ConceptType.METHOD,
    'simulation': ConceptType.METHOD,
    'algorithm': ConceptType.ALGORITHM,
    'heuristic': ConceptType.ALGORITHM,
    'optimization': ConceptType.METHOD,
    'benchmark': ConceptType.METHOD,
    
    # Phase 1+2: Security/Crypto Actors (common in protocol papers)
    'alice': ConceptType.PERSON,
    'bob': ConceptType.PERSON,
    'charlie': ConceptType.PERSON,
    'david': ConceptType.PERSON,
    'eve': ConceptType.PERSON,
    'mallory': ConceptType.PERSON,
    'trudy': ConceptType.PERSON,
    'attacker': ConceptType.ACTOR,
    'adversary': ConceptType.ACTOR,
    'intruder': ConceptType.ACTOR,
    'eavesdropper': ConceptType.ACTOR,
    'verifier': ConceptType.ROLE,
    'prover': ConceptType.ROLE,
    'challenger': ConceptType.ROLE,
    'participant': ConceptType.ROLE,
    'player': ConceptType.ROLE,
    'user': ConceptType.ROLE,
    
    # Phase 1+2: Cryptography & Security
    'cryptography': ConceptType.CRYPTOGRAPHY,
    'cryptographic': ConceptType.CRYPTOGRAPHY,
    'cipher': ConceptType.CRYPTOGRAPHY,
    'hash': ConceptType.CRYPTOGRAPHY,
    'signature': ConceptType.CRYPTOGRAPHY,
    'key': ConceptType.CRYPTOGRAPHY,
    'certificate': ConceptType.CRYPTOGRAPHY,
    'secret': ConceptType.CRYPTOGRAPHY,
    
    # Phase 1+2: AI Agent Frameworks
    'langgraph': ConceptType.FRAMEWORK,
    'autogen': ConceptType.FRAMEWORK,
    'crewai': ConceptType.FRAMEWORK,
    'metagpt': ConceptType.FRAMEWORK,
    'agora': ConceptType.FRAMEWORK,
    'swarm': ConceptType.FRAMEWORK,
    'haystack': ConceptType.FRAMEWORK,
    'langchain': ConceptType.FRAMEWORK,
    
    # Phase 1+2: Formal Methods & Verification Tools
    'cpsa': ConceptType.TOOL,
    'uppaal': ConceptType.TOOL,
    'proverif': ConceptType.TOOL,
    'tamarin': ConceptType.TOOL,
    'coq': ConceptType.TOOL,
    'isabelle': ConceptType.TOOL,
    'proof': ConceptType.FORMAL_METHOD,
    'verification': ConceptType.FORMAL_METHOD,
    'theorem': ConceptType.FORMAL_METHOD,
    'lemma': ConceptType.FORMAL_METHOD,
    'invariant': ConceptType.FORMAL_METHOD,
    'induction': ConceptType.FORMAL_METHOD,
    
    # Phase 1+2: Network/Distributed Systems
    'gossip': ConceptType.PROTOCOL,
    'tcp': ConceptType.PROTOCOL,
    'udp': ConceptType.PROTOCOL,
    'gridftp': ConceptType.PROTOCOL,
    'ftp': ConceptType.PROTOCOL,
    'smtp': ConceptType.PROTOCOL,
    'client': ConceptType.ROLE,
    'server': ConceptType.ROLE,
    'coordinator': ConceptType.ROLE,
    'node': ConceptType.COMPONENT,
    'peer': ConceptType.COMPONENT,
    'router': ConceptType.COMPONENT,
    'gateway': ConceptType.COMPONENT,
    
    # Phase 1+2: Distributed Systems Concepts
    'consensus': ConceptType.DISTRIBUTED_SYSTEM,
    'replication': ConceptType.DISTRIBUTED_SYSTEM,
    'synchronization': ConceptType.DISTRIBUTED_SYSTEM,
    'consistency': ConceptType.DISTRIBUTED_SYSTEM,
    'partition': ConceptType.DISTRIBUTED_SYSTEM,
    'sharding': ConceptType.DISTRIBUTED_SYSTEM,
    'fault tolerance': ConceptType.DISTRIBUTED_SYSTEM,
    
    # Phase 1+2: Organizations & Companies
    'google': ConceptType.COMPANY,
    'openai': ConceptType.COMPANY,
    'anthropic': ConceptType.COMPANY,
    'microsoft': ConceptType.COMPANY,
    'meta': ConceptType.COMPANY,
    'amazon': ConceptType.COMPANY,
    'apple': ConceptType.COMPANY,
    'nvidia': ConceptType.COMPANY,
    'university': ConceptType.ORGANIZATION,
    'institute': ConceptType.ORGANIZATION,
    'laboratory': ConceptType.ORGANIZATION,
    'foundation': ConceptType.ORGANIZATION,
    
    # Phase 1+2: Common Technical Terms
    'implementation': ConceptType.METHOD,
    'design': ConceptType.METHOD,
    'deployment': ConceptType.METHOD,
    'execution': ConceptType.METHOD,
    'performance': ConceptType.METRIC,
    'efficiency': ConceptType.METRIC,
    'throughput': ConceptType.METRIC,
    'latency': ConceptType.METRIC,
    'scalability': ConceptType.METRIC,
    'reliability': ConceptType.METRIC,
    'availability': ConceptType.METRIC,
    
    # Phase 1+2: Components & Roles
    'sender': ConceptType.ROLE,
    'receiver': ConceptType.ROLE,
    'initiator': ConceptType.ROLE,
    'responder': ConceptType.ROLE,
    'observer': ConceptType.ROLE,
    'monitor': ConceptType.COMPONENT,
    'controller': ConceptType.COMPONENT,
    'manager': ConceptType.COMPONENT,
    'handler': ConceptType.COMPONENT,
    'adapter': ConceptType.COMPONENT,
    'wrapper': ConceptType.COMPONENT,
    'proxy': ConceptType.COMPONENT,
    
    # Phase 1+2: Resources & Data
    'resource': ConceptType.DATA,
    'file': ConceptType.DATA,
    'database': ConceptType.SYSTEM,
    'storage': ConceptType.SYSTEM,
    'cache': ConceptType.SYSTEM,
    'buffer': ConceptType.DATA,
    'queue': ConceptType.DATA,
    'stream': ConceptType.DATA,
    
    # Phase 1+2: Development & Tools
    'tool': ConceptType.TOOL,
    'library': ConceptType.FRAMEWORK,
    'package': ConceptType.FRAMEWORK,
    'module': ConceptType.COMPONENT,
    'plugin': ConceptType.COMPONENT,
    'extension': ConceptType.COMPONENT,
    'compiler': ConceptType.TOOL,
    'interpreter': ConceptType.TOOL,
    'debugger': ConceptType.TOOL,
    
    # Phase 1+2: Environments & Platforms
    'environment': ConceptType.SYSTEM,
    'platform': ConceptType.SYSTEM,
    'container': ConceptType.SYSTEM,
    'virtual machine': ConceptType.SYSTEM,
    'cloud': ConceptType.SYSTEM,
    'edge': ConceptType.SYSTEM,
    
    # Phase 1+2: Processes & Tasks
    'process': ConceptType.PROCESS,
    'task': ConceptType.PROCESS,
    'job': ConceptType.PROCESS,
    'thread': ConceptType.PROCESS,
    'workflow': ConceptType.PROCESS,
    'pipeline': ConceptType.PROCESS,
    
    # Phase 1+2: Information & Knowledge
    'information': ConceptType.DATA,
    'knowledge': ConceptType.KNOWLEDGE,
    'ontology': ConceptType.KNOWLEDGE_GRAPH,
    'schema': ConceptType.DATA,
    'specification': ConceptType.DATA,
    
    # Phase 1+2: Interaction & Communication
    'request': ConceptType.MESSAGE,
    'response': ConceptType.MESSAGE,
    'query': ConceptType.MESSAGE,
    'command': ConceptType.MESSAGE,
    'event': ConceptType.MESSAGE,
    'notification': ConceptType.MESSAGE,
    'signal': ConceptType.MESSAGE,
    
    # Phase 1+2: Common Acronyms & Abbreviations
    'acp': ConceptType.PROTOCOL,
    'ecp': ConceptType.PROTOCOL,
    'mas': ConceptType.SYSTEM,
    'dht': ConceptType.NETWORK,
    'p2p': ConceptType.NETWORK,
    'rpc': ConceptType.PROTOCOL,
    'grpc': ConceptType.PROTOCOL,
    'soap': ConceptType.PROTOCOL,
}


class GraphManager:
    """Manage knowledge graph for enhanced context retrieval"""
    
    def __init__(self):
        log.info("Initializing GraphManager...")
        
        # Initialize ontology loader for classification
        self.ontology = get_ontology_loader()
        log.info("✓ Loaded graph ontology configuration")
        
        # Initialize metadata database
        try:
            self.metadata_db = get_metadata_db()
            log.info("✓ Metadata database initialized in GraphManager")
        except Exception as e:
            log.warning(f"Metadata database initialization failed: {e}")
            self.metadata_db = None
        
        # Initialize graph store
        self.graph_store_path = settings.GRAPH_STORE_PATH / "graph_store.json"
        self.graph_store = SimpleGraphStore()
        
        # NetworkX graph for analysis and visualization
        self.nx_graph = nx.DiGraph()
        
        # Load existing graph if available
        if self.graph_store_path.exists():
            try:
                self.graph_store = SimpleGraphStore.from_persist_path(
                    str(self.graph_store_path)
                )
                log.info("✓ Loaded existing graph store")
                # Sync loaded graph to NetworkX for visualization and stats
                self._sync_to_networkx()
            except Exception as e:
                log.warning(f"Could not load graph store: {e}")
                self.graph_store = SimpleGraphStore()
        
        # Create storage context with graph store
        self.storage_context = StorageContext.from_defaults(
            graph_store=self.graph_store
        )
        
        # Initialize knowledge graph index (will be set during build)
        self.kg_index = None
        
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
    
    def extract_concepts_from_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract research concepts from text using keyword matching.
        
        This method now uses the OntologyLoader to access keyword mappings
        from config/graph_ontology.yaml.
        
        Args:
            text: Text to extract concepts from
            
        Returns:
            List of (keyword, concept_type_name) tuples
        """
        concepts = []
        text_lower = text.lower()
        
        # Get keyword mappings from ontology
        keyword_map = self.ontology.get_keyword_mappings()
        
        for keyword, concept_type in keyword_map.items():
            if keyword in text_lower:
                concepts.append((keyword, concept_type))
        
        return concepts
    
    def _classify_node(self, node_label: str) -> str:
        """
        Classify a node based on its label using ontology configuration.
        
        This method now uses the OntologyLoader to enable configuration-based
        classification without code changes. The classification logic includes
        both keyword matching and pattern-based rules defined in
        config/graph_ontology.yaml.
        
        Args:
            node_label: The label/name of the node
            
        Returns:
            String representation of concept type or 'unknown'
        """
        return self.ontology.classify_node(node_label)
    
    def _normalize_entity_name(self, entity: str) -> str:
        """
        Normalize entity name for better matching.
        
        Args:
            entity: Entity name to normalize
            
        Returns:
            Normalized entity name
        """
        # Convert to lowercase
        normalized = entity.lower().strip()
        
        # Common abbreviation expansions
        abbreviations = {
            'a2a': 'agent2agent',
            'mcp': 'model context protocol',
            'anp': 'agent network protocol',
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
        }
        
        # Check if it's a known abbreviation
        if normalized in abbreviations:
            normalized = abbreviations[normalized]
        
        return normalized
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings using multiple methods.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        str1_norm = self._normalize_entity_name(str1)
        str2_norm = self._normalize_entity_name(str2)
        
        # Exact match after normalization
        if str1_norm == str2_norm:
            return 1.0
        
        # Jaccard similarity (token-based)
        tokens1 = set(str1_norm.split())
        tokens2 = set(str2_norm.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Substring match bonus
        substring_bonus = 0.0
        if str1_norm in str2_norm or str2_norm in str1_norm:
            substring_bonus = 0.3
        
        return min(jaccard + substring_bonus, 1.0)
    
    def merge_similar_nodes(self, similarity_threshold: float = 0.7) -> Dict:
        """
        Merge nodes that are likely referring to the same entity.
        
        Args:
            similarity_threshold: Minimum similarity score to merge nodes
            
        Returns:
            Dictionary with merge statistics
        """
        try:
            log.info(f"Starting entity resolution with threshold {similarity_threshold}")
            
            nodes = list(self.nx_graph.nodes())
            merged_count = 0
            merge_groups = []
            processed = set()
            
            for i, node1 in enumerate(nodes):
                if node1 in processed:
                    continue
                
                similar_nodes = [node1]
                
                for node2 in nodes[i+1:]:
                    if node2 in processed:
                        continue
                    
                    similarity = self._calculate_similarity(node1, node2)
                    
                    if similarity >= similarity_threshold:
                        similar_nodes.append(node2)
                        processed.add(node2)
                
                if len(similar_nodes) > 1:
                    merge_groups.append(similar_nodes)
                    # Merge all similar nodes into the first one
                    primary_node = similar_nodes[0]
                    
                    for node in similar_nodes[1:]:
                        # Transfer all edges from node to primary_node
                        for neighbor in list(self.nx_graph.neighbors(node)):
                            edge_data = self.nx_graph[node][neighbor]
                            if not self.nx_graph.has_edge(primary_node, neighbor):
                                self.nx_graph.add_edge(
                                    primary_node,
                                    neighbor,
                                    **edge_data
                                )
                        
                        # Transfer incoming edges
                        for predecessor in list(self.nx_graph.predecessors(node)):
                            edge_data = self.nx_graph[predecessor][node]
                            if not self.nx_graph.has_edge(predecessor, primary_node):
                                self.nx_graph.add_edge(
                                    predecessor,
                                    primary_node,
                                    **edge_data
                                )
                        
                        # Merge metadata
                        if 'frequency' in self.nx_graph.nodes[node]:
                            primary_freq = self.nx_graph.nodes[primary_node].get('frequency', 0)
                            node_freq = self.nx_graph.nodes[node].get('frequency', 0)
                            self.nx_graph.nodes[primary_node]['frequency'] = primary_freq + node_freq
                        
                        # Remove the duplicate node
                        self.nx_graph.remove_node(node)
                        merged_count += 1
                
                processed.add(node1)
            
            log.info(f"✓ Merged {merged_count} duplicate nodes into {len(merge_groups)} entities")
            
            return {
                "merged_count": merged_count,
                "merge_groups": merge_groups,
                "remaining_nodes": len(self.nx_graph.nodes),
                "remaining_edges": len(self.nx_graph.edges)
            }
            
        except Exception as e:
            log.error(f"Error merging similar nodes: {e}")
            return {
                "merged_count": 0,
                "error": str(e)
            }
    
    def _normalize_relationship(self, relationship: str) -> str:
        """
        Normalize relationship to standard type using ontology configuration.
        
        This method now uses the OntologyLoader to enable configuration-based
        relationship normalization without code changes. Normalization rules
        are defined in config/graph_ontology.yaml.
        
        Args:
            relationship: Raw relationship string
            
        Returns:
            Normalized relationship name
        """
        return self.ontology.normalize_relationship(relationship)
    
    def normalize_all_relationships(self) -> Dict:
        """
        Normalize all relationships in the graph to standard types.
        
        Returns:
            Dictionary with normalization statistics
        """
        try:
            log.info("Normalizing relationship types...")
            
            normalized_count = 0
            original_types = set()
            normalized_types = set()
            
            # Create a new graph with normalized relationships
            edges_to_update = []
            
            for source, target, data in self.nx_graph.edges(data=True):
                original_rel = data.get('relationship', 'related_to')
                normalized_rel = self._normalize_relationship(original_rel)
                
                original_types.add(original_rel)
                normalized_types.add(normalized_rel)
                
                if original_rel != normalized_rel:
                    edges_to_update.append((source, target, normalized_rel))
                    normalized_count += 1
            
            # Update relationships
            for source, target, new_rel in edges_to_update:
                self.nx_graph[source][target]['relationship'] = new_rel
            
            log.info(f"✓ Normalized {normalized_count} relationships")
            log.info(f"  Original types: {len(original_types)}")
            log.info(f"  Normalized types: {len(normalized_types)}")
            
            return {
                "normalized_count": normalized_count,
                "original_type_count": len(original_types),
                "normalized_type_count": len(normalized_types),
                "reduction_percentage": ((len(original_types) - len(normalized_types)) / len(original_types) * 100) if original_types else 0
            }
            
        except Exception as e:
            log.error(f"Error normalizing relationships: {e}")
            return {
                "normalized_count": 0,
                "error": str(e)
            }
    
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
            # Graph store needs a file path, not a directory
            graph_file = settings.GRAPH_STORE_PATH / "graph_store.json"
            self.graph_store.persist(
                persist_path=str(graph_file)
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
            
            # Graph store needs a file path, not a directory
            graph_file = settings.GRAPH_STORE_PATH / "graph_store.json"
            if not graph_file.exists():
                log.info("No existing graph store file found")
                return False
            
            self.graph_store = SimpleGraphStore.from_persist_path(
                str(graph_file)
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
            
            # Access the internal graph_dict from SimpleGraphStore
            # SimpleGraphStore stores data in _data.graph_dict where:
            # - keys are subject nodes
            # - values are lists of [relation, object] pairs
            if hasattr(self.graph_store, '_data') and hasattr(self.graph_store._data, 'graph_dict'):
                graph_dict = self.graph_store._data.graph_dict
                
                # Track node metadata
                node_metadata = {}  # {node_name: {sources: set(), first_seen: str, count: int}}
                
                # First pass: add all edges and collect metadata
                for subject, relations in graph_dict.items():
                    # Initialize subject metadata
                    if subject not in node_metadata:
                        node_metadata[subject] = {
                            'sources': set(),
                            'first_seen': datetime.now().isoformat(),
                            'frequency': 0
                        }
                    node_metadata[subject]['frequency'] += len(relations)
                    
                    for relation_obj_pair in relations:
                        # Each pair is [relation, object]
                        if len(relation_obj_pair) == 2:
                            rel, obj = relation_obj_pair
                            
                            # Initialize object metadata
                            if obj not in node_metadata:
                                node_metadata[obj] = {
                                    'sources': set(),
                                    'first_seen': datetime.now().isoformat(),
                                    'frequency': 0
                                }
                            node_metadata[obj]['frequency'] += 1
                            
                            self.nx_graph.add_edge(
                                subject,
                                obj,
                                relationship=rel
                            )
                
                # Second pass: classify and assign types + metadata to all nodes
                for node in self.nx_graph.nodes():
                    node_type = self._classify_node(node)
                    metadata = node_metadata.get(node, {
                        'sources': set(),
                        'first_seen': datetime.now().isoformat(),
                        'frequency': 0
                    })
                    
                    self.nx_graph.nodes[node]['type'] = node_type
                    self.nx_graph.nodes[node]['frequency'] = metadata['frequency']
                    self.nx_graph.nodes[node]['first_seen'] = metadata['first_seen']
                    
                    # Save node to metadata database
                    if self.metadata_db is not None:
                        try:
                            # Generate unique node ID from label
                            node_id = hashlib.sha256(node.encode()).hexdigest()[:16]
                            
                            self.metadata_db.insert_graph_node(
                                node_id=node_id,
                                label=node,
                                node_type=node_type,
                                classification_method='pattern',
                                frequency=metadata['frequency'],
                                first_seen=metadata['first_seen']
                            )
                        except Exception as e:
                            log.debug(f"Could not save node to DB: {e}")
                
                # Third pass: save relationships to database
                if self.metadata_db is not None:
                    try:
                        for source, target, data in self.nx_graph.edges(data=True):
                            # Generate unique IDs
                            source_id = hashlib.sha256(source.encode()).hexdigest()[:16]
                            target_id = hashlib.sha256(target.encode()).hexdigest()[:16]
                            rel_type = data.get('relationship', 'related_to')
                            
                            # Create relationship ID
                            rel_hash = f"{source_id}_{rel_type}_{target_id}"
                            relationship_id = hashlib.sha256(rel_hash.encode()).hexdigest()[:16]
                            
                            normalized_type = self._normalize_relationship(rel_type)
                            
                            self.metadata_db.insert_relationship(
                                relationship_id=relationship_id,
                                source_node_id=source_id,
                                target_node_id=target_id,
                                relationship_type=rel_type,
                                normalized_type=normalized_type
                            )
                    except Exception as e:
                        log.debug(f"Could not save relationships to DB: {e}")
            
            log.info(f"✓ Synced to NetworkX: {len(self.nx_graph.nodes)} nodes, {len(self.nx_graph.edges)} edges")
            
        except Exception as e:
            log.error(f"Error syncing to NetworkX: {e}")
    
    def reclassify_all_nodes(self) -> Dict[str, int]:
        """
        Reclassify all nodes in the graph using updated classification logic.
        
        Phase 1+2: This method reclassifies all existing nodes to take
        advantage of the expanded CONCEPT_KEYWORDS and pattern matching.
        
        Returns:
            Dict with classification statistics
        """
        try:
            log.info("Reclassifying all nodes with updated logic...")
            
            stats = {
                'total_nodes': len(self.nx_graph.nodes),
                'reclassified': 0,
                'unchanged': 0,
                'changes': {}  # old_type -> new_type -> count
            }
            
            for node in self.nx_graph.nodes():
                old_type = self.nx_graph.nodes[node].get('type', 'unknown')
                new_type = self._classify_node(node)
                
                if old_type != new_type:
                    stats['reclassified'] += 1
                    change_key = f"{old_type} -> {new_type}"
                    stats['changes'][change_key] = (
                        stats['changes'].get(change_key, 0) + 1
                    )
                    self.nx_graph.nodes[node]['type'] = new_type
                else:
                    stats['unchanged'] += 1
            
            log.info(
                f"✓ Reclassification complete: "
                f"{stats['reclassified']} nodes reclassified, "
                f"{stats['unchanged']} unchanged"
            )
            
            return stats
            
        except Exception as e:
            log.error(f"Error reclassifying nodes: {e}")
            return {'error': str(e)}
    
    def reload_ontology(self) -> Dict:
        """
        Reload ontology configuration from disk and optionally reclassify.
        
        This allows dynamic updates to classification rules without
        restarting the application.
        
        Returns:
            Dict with reload statistics
        """
        try:
            log.info("Reloading ontology configuration...")
            
            # Reload the configuration
            from config.ontology_loader import reload_ontology
            reload_ontology()
            
            # Get fresh reference
            self.ontology = get_ontology_loader()
            
            stats = self.ontology.get_statistics()
            
            log.info("✓ Ontology configuration reloaded")
            log.info(f"  Concept types: {stats['concept_types']}")
            log.info(f"  Keywords: {stats['keyword_mappings']}")
            log.info(f"  Patterns: {stats['classification_patterns']}")
            
            return {
                'reloaded': True,
                'stats': stats
            }
            
        except Exception as e:
            log.error(f"Error reloading ontology: {e}")
            return {
                'reloaded': False,
                'error': str(e)
            }
    
    def reclassify_with_hybrid(
        self,
        batch_size: int = 100,
        dry_run: bool = False,
        use_cache: bool = True
    ) -> Dict:
        """
        Reclassify unknown nodes using hybrid approach:
        1. Pattern/keyword matching (already applied)
        2. Non-concept filtering (timestamps, paths, etc.)
        3. LLM-based classification for remaining unknowns
        
        Args:
            batch_size: Number of nodes to process in each LLM batch
            dry_run: If True, only report what would be done
            use_cache: Whether to use classification cache
            
        Returns:
            Dict with reclassification statistics
        """
        try:
            from config.settings import Settings
            
            log.info("Starting hybrid classification...")
            
            # Collect unknown nodes
            unknown_nodes = []
            for node_id, attrs in self.nx_graph.nodes(data=True):
                if attrs.get('type', 'unknown') == 'unknown':
                    label = attrs.get('label', str(node_id))
                    unknown_nodes.append((node_id, label))
            
            total_unknown = len(unknown_nodes)
            log.info(f"Found {total_unknown} unknown nodes")
            
            if total_unknown == 0:
                return {
                    'total_processed': 0,
                    'non_concepts': 0,
                    'llm_classified': 0,
                    'still_unknown': 0
                }
            
            # Phase 1: Filter non-concepts
            non_concepts = []
            remaining_unknowns = []
            
            for node_id, label in unknown_nodes:
                if self.ontology.is_non_concept(label):
                    non_concepts.append((node_id, label))
                else:
                    remaining_unknowns.append((node_id, label))
            
            log.info(f"Filtered {len(non_concepts)} non-concepts")
            log.info(
                f"Remaining for LLM classification: {len(remaining_unknowns)}"
            )
            
            # Phase 2: LLM classification (with batching)
            llm_classified = 0
            still_unknown = 0
            
            if not dry_run and remaining_unknowns:
                log.info(
                    f"Starting LLM classification in batches of {batch_size}..."
                )
                
                for i in range(0, len(remaining_unknowns), batch_size):
                    batch = remaining_unknowns[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    total_batches = (
                        len(remaining_unknowns) + batch_size - 1
                    ) // batch_size
                    
                    log.info(
                        f"Processing batch {batch_num}/{total_batches} "
                        f"({len(batch)} nodes)..."
                    )
                    
                    for node_id, label in batch:
                        # Get context for better classification
                        context = self._get_node_context(node_id)
                        
                        # Classify with LLM
                        classification = self.ontology.classify_with_llm(
                            label,
                            context=context,
                            llm_provider=Settings.llm,
                            use_cache=use_cache
                        )
                        
                        if classification != 'unknown':
                            self.nx_graph.nodes[node_id]['type'] = classification
                            llm_classified += 1
                        else:
                            still_unknown += 1
                    
                    # Progress update
                    log.info(
                        f"  Classified: {llm_classified}, "
                        f"Unknown: {still_unknown}"
                    )
            
            result = {
                'total_processed': total_unknown,
                'non_concepts': len(non_concepts),
                'llm_candidates': len(remaining_unknowns),
                'llm_classified': llm_classified,
                'still_unknown': still_unknown,
                'dry_run': dry_run
            }
            
            if dry_run:
                log.info("Dry run complete - no changes made")
            else:
                log.info("✓ Hybrid classification complete")
                log.info(f"  Non-concepts filtered: {len(non_concepts)}")
                log.info(f"  LLM classified: {llm_classified}")
                log.info(f"  Still unknown: {still_unknown}")
            
            return result
            
        except Exception as e:
            log.error(f"Error in hybrid classification: {e}")
            return {'error': str(e)}
    
    def _get_node_context(self, node_id: str) -> Dict:
        """
        Get context for a node to help with LLM classification.
        
        Args:
            node_id: The node ID
            
        Returns:
            Dict with neighbors, edge_types, etc.
        """
        context = {
            'neighbors': [],
            'edge_types': []
        }
        
        try:
            # Get neighbor labels
            for neighbor in self.nx_graph.neighbors(node_id):
                neighbor_label = self.nx_graph.nodes[neighbor].get(
                    'label',
                    neighbor
                )
                context['neighbors'].append(neighbor_label)
            
            # Get edge types
            for _, _, edge_data in self.nx_graph.edges(node_id, data=True):
                rel = edge_data.get('relationship', 'related_to')
                context['edge_types'].append(rel)
            
        except Exception as e:
            log.debug(f"Error getting context for {node_id}: {e}")
        
        return context
    
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
