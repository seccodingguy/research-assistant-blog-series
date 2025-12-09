# agents/research_assistant_agent.py
from typing import Optional, Dict, List, Any
from pathlib import Path
import os
import logging
from agents.pdf_agent import PDFAgent
from utils.logger import log
from utils.exceptions import (
    DocumentProcessingError,
    InvalidDocumentError,
    DocumentNotFoundError,
    SearchError,
    InvalidQueryError,
    PathValidationError
)
from config import settings

logfilename = os.path.join("logs", "research_assistant_agent.log")

# Configure logging
logging.basicConfig(
    filename=logfilename,
    level=logging.INFO,  # Changed from DEBUG to INFO to prevent log spam
    filemode="a",
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
FileOutputHandler = logging.FileHandler(logfilename)

logger.addHandler(FileOutputHandler)

# Suppress third-party DEBUG logging to prevent massive log files
logging.getLogger("pdfminer").setLevel(logging.WARNING)

class ResearchAssistantAgent:
    """
    High-level research assistant that orchestrates PDF processing,
    paper search, knowledge graph queries, conversation management,
    and complex multi-step research workflows.
    """
    
    def __init__(self, user_id: str = "default", auto_watch: Optional[bool] = None):
        """Initialize the research assistant agent"""
        log.info("Initializing ResearchAssistantAgent...")
        
        self.user_id = user_id
        
        # Initialize PDF agent for document processing and search
        self.pdf_agent = PDFAgent(user_id=user_id, auto_watch=auto_watch)
        
        # Paper search service (lazy initialization)
        self._paper_service = None
        self._paper_service_initialized = False
        
        # Workflow orchestrator (lazy initialization)
        self._workflow_orchestrator = None
        
        log.info("ResearchAssistantAgent initialized successfully!")
    
    async def _ensure_paper_service(self):
        """Ensure paper search service is initialized (async-safe)"""
        if self._paper_service is None:
            from services.paper_search_service import PaperSearchService
            self._paper_service = PaperSearchService()
        
        if not self._paper_service_initialized:
            await self._paper_service.initialize()
            self._paper_service_initialized = True
        
        return self._paper_service
    
    def _get_workflow_orchestrator(self):
        """Get or create workflow orchestrator (lazy initialization)"""
        if self._workflow_orchestrator is None:
            from core.workflow_orchestrator import WorkflowOrchestrator
            self._workflow_orchestrator = WorkflowOrchestrator(self)
            log.info("Workflow orchestrator initialized")
        return self._workflow_orchestrator
    
    # ==================== Document Processing ====================
    
    def process_pdf(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single PDF file
        
        Returns:
            Dict with 'success' (bool) and 'message' (str)
        """
        try:
            # Validate file path to prevent directory traversal
            file_path = Path(file_path).resolve()
            if not file_path.exists():
                return {
                    "success": False,
                    "message": f"File not found: {file_path.name}",
                    "file_name": str(file_path.name)
                }
            
            if not file_path.is_file():
                return {
                    "success": False,
                    "message": f"Path is not a file: {file_path.name}",
                    "file_name": str(file_path.name)
                }
            
            if not file_path.suffix.lower() == '.pdf':
                return {
                    "success": False,
                    "message": f"Not a PDF file: {file_path.name}",
                    "file_name": file_path.name
                }
            
            success = self.pdf_agent.process_pdf(file_path)
            return {
                "success": success,
                "message": f"Successfully processed: {file_path.name}" if success else f"Failed to process: {file_path.name}",
                "file_name": file_path.name
            }
        except Exception as e:
            log.exception(f"Error processing PDF: {file_path}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "file_name": file_path.name
            }
    
    def process_folder(self, folder_path: Path) -> Dict[str, Any]:
        """
        Process all PDFs in a folder
        
        Returns:
            Dict with 'success', 'documents_processed', and optional 'error'
        """
        try:
            # Validate folder path
            folder_path = Path(folder_path).resolve()
            if not folder_path.exists():
                raise DocumentNotFoundError(f"Folder not found: {folder_path}")
            
            if not folder_path.is_dir():
                raise PathValidationError(f"Path is not a directory: {folder_path}")
            
            result = self.pdf_agent.process_folder(folder_path)
            return result
        except (DocumentNotFoundError, PathValidationError) as e:
            log.error(f"Validation error processing folder: {folder_path} - {e}")
            return {
                "success": False,
                "error": str(e),
                "documents_processed": 0
            }
        except DocumentProcessingError as e:
            log.error(f"Processing error for folder: {folder_path} - {e}")
            return {
                "success": False,
                "error": str(e),
                "documents_processed": 0
            }
        except Exception as e:
            log.exception(f"Unexpected error processing folder: {folder_path}")
            return {
                "success": False,
                "error": str(e),
                "documents_processed": 0
            }
    
    # ==================== Search & Chat ====================
    
    def search(self, query: str, mode: str = "enhanced", save_to_memory: bool = True) -> Dict[str, Any]:
        """
        Search the knowledge base
        
        Args:
            query: Search query
            mode: Search mode (basic, enhanced, analyze_all)
            save_to_memory: Whether to save to conversation memory
        
        Returns:
            Dict with 'answer', 'sources', 'metadata'
        """
        try:
            # Validate input
            if not query or not query.strip():
                return {
                    "answer": "Please provide a valid search query.",
                    "sources": [],
                    "error": "Empty query"
                }
            
            query = query.strip()
            
            # Validate query length
            if len(query) > 5000:
                return {
                    "answer": "Query too long. Please limit to 5000 characters.",
                    "sources": [],
                    "error": "Query too long"
                }
            
            # Validate mode
            valid_modes = ["basic", "enhanced", "analyze_all"]
            if mode not in valid_modes:
                mode = "enhanced"  # Default to safe mode
            
            return self.pdf_agent.search(query, mode=mode, save_to_memory=save_to_memory)
        except Exception as e:
            log.exception(f"Error during search: {query}")
            return {
                "answer": f"Search failed: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def chat(self, message: str, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Interactive chat with context awareness
        
        Args:
            message: User message
            mode: Optional mode override
        
        Returns:
            Dict with 'response' and 'metadata'
        """
        try:
            response = self.pdf_agent.chat(message, mode=mode)
            
            # Get metadata from search (without saving to memory again)
            result = self.pdf_agent.search(message, mode=mode or "enhanced", save_to_memory=False)
            
            return {
                "response": response,
                "sources": result.get("sources", []),
                "metadata": result.get("metadata", {})
            }
        except SearchError as e:
            log.error(f"Search error during chat: {message} - {e}")
            return {
                "response": f"Search failed: {str(e)}",
                "sources": [],
                "error": str(e)
            }
        except Exception as e:
            log.exception(f"Unexpected error during chat: {message}")
            return {
                "response": f"Chat failed: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def get_sources(self, query: str) -> List[Dict[str, Any]]:
        """Get source documents for a query"""
        try:
            return self.pdf_agent.get_sources(query)
        except Exception as e:
            log.exception(f"Error getting sources: {query}")
            return []
    
    # ==================== Knowledge Graph ====================
    
    def get_graph_stats(self) -> Optional[Dict[str, Any]]:
        """Get knowledge graph statistics"""
        try:
            if hasattr(self.pdf_agent, 'pdf_parser') and hasattr(self.pdf_agent.pdf_parser, 'graph_manager'):
                graph_mgr = self.pdf_agent.pdf_parser.graph_manager
                if graph_mgr is not None:
                    return graph_mgr.get_graph_statistics()
            return None
        except Exception as e:
            log.exception("Error getting graph stats")
            return None
    
    def visualize_graph(self, output_path: Path, max_nodes: int = 100, query: Optional[str] = None) -> Dict[str, Any]:
        """
        Visualize knowledge graph
        
        Args:
            output_path: Path to save visualization
            max_nodes: Maximum nodes to include
            query: Optional query to filter graph
        
        Returns:
            Dict with visualization data
        """
        try:
            if hasattr(self.pdf_agent, 'pdf_parser') and hasattr(self.pdf_agent.pdf_parser, 'graph_manager'):
                graph_mgr = self.pdf_agent.pdf_parser.graph_manager
                if graph_mgr is not None:
                    viz_data = graph_mgr.visualize_graph(
                        output_path=output_path,
                        max_nodes=max_nodes
                    )
                    return {
                        "success": True,
                        "data": viz_data,
                        "output_path": str(output_path)
                    }
            return {
                "success": False,
                "error": "Knowledge graph not available"
            }
        except Exception as e:
            log.exception("Error visualizing graph")
            return {
                "success": False,
                "error": str(e)
            }
    
    def query_graph(self, query: str) -> Dict[str, Any]:
        """
        Query the knowledge graph
        
        Args:
            query: Query string
        
        Returns:
            Dict with response, nodes, and relationships
        """
        try:
            if hasattr(self.pdf_agent, 'pdf_parser') and hasattr(self.pdf_agent.pdf_parser, 'graph_manager'):
                graph_mgr = self.pdf_agent.pdf_parser.graph_manager
                if graph_mgr is not None:
                    result = graph_mgr.query_graph(query)
                    return {
                        "success": True,
                        "response": result.get('response', ''),
                        "nodes": result.get('nodes', []),
                        "relationships": result.get('relationships', [])
                    }
            return {
                "success": False,
                "error": "Knowledge graph not available"
            }
        except Exception as e:
            log.exception(f"Error querying graph: {query}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def merge_graph_entities(self, similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Merge similar entities in the knowledge graph.
        
        Args:
            similarity_threshold: Minimum similarity to merge (0.0-1.0)
        
        Returns:
            Dict with merge statistics
        """
        try:
            if hasattr(self.pdf_agent, 'pdf_parser') and hasattr(self.pdf_agent.pdf_parser, 'graph_manager'):
                graph_mgr = self.pdf_agent.pdf_parser.graph_manager
                if graph_mgr is not None:
                    result = graph_mgr.merge_similar_nodes(similarity_threshold)
                    return {
                        "success": True,
                        **result
                    }
            return {
                "success": False,
                "error": "Knowledge graph not available"
            }
        except Exception as e:
            log.exception("Error merging graph entities")
            return {
                "success": False,
                "error": str(e)
            }
    
    def normalize_graph_relationships(self) -> Dict[str, Any]:
        """
        Normalize relationship types in the knowledge graph.
        
        Returns:
            Dict with normalization statistics
        """
        try:
            if hasattr(self.pdf_agent, 'pdf_parser') and hasattr(self.pdf_agent.pdf_parser, 'graph_manager'):
                graph_mgr = self.pdf_agent.pdf_parser.graph_manager
                if graph_mgr is not None:
                    result = graph_mgr.normalize_all_relationships()
                    return {
                        "success": True,
                        **result
                    }
            return {
                "success": False,
                "error": "Knowledge graph not available"
            }
        except Exception as e:
            log.exception("Error normalizing relationships")
            return {
                "success": False,
                "error": str(e)
            }
    
    def reclassify_graph_nodes(self) -> Dict[str, Any]:
        """
        Reclassify all nodes in the knowledge graph using updated logic.
        
        Phase 1+2: Applies expanded CONCEPT_KEYWORDS and pattern matching
        to reclassify existing nodes.
        
        Returns:
            Dict with reclassification statistics
        """
        try:
            if hasattr(self.pdf_agent, 'pdf_parser') and hasattr(
                self.pdf_agent.pdf_parser, 'graph_manager'
            ):
                graph_mgr = self.pdf_agent.pdf_parser.graph_manager
                if graph_mgr is not None:
                    result = graph_mgr.reclassify_all_nodes()
                    return {
                        "success": True,
                        **result
                    }
            return {
                "success": False,
                "error": "Knowledge graph not available"
            }
        except Exception as e:
            log.exception("Error reclassifying nodes")
            return {
                "success": False,
                "error": str(e)
            }
    
    def reclassify_hybrid(
        self,
        batch_size: int = 100,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Reclassify unknown nodes using hybrid approach:
        1. Pattern/keyword matching
        2. Non-concept filtering (timestamps, paths, variables)
        3. LLM-based classification for remaining unknowns
        
        Phase 4: Hybrid approach to improve classification accuracy.
        
        Args:
            batch_size: Number of nodes to process per LLM batch
            dry_run: If True, only report what would be done
            
        Returns:
            Dict with hybrid classification statistics
        """
        try:
            if hasattr(self.pdf_agent, 'pdf_parser') and hasattr(
                self.pdf_agent.pdf_parser, 'graph_manager'
            ):
                graph_mgr = self.pdf_agent.pdf_parser.graph_manager
                if graph_mgr is not None:
                    result = graph_mgr.reclassify_with_hybrid(
                        batch_size=batch_size,
                        dry_run=dry_run
                    )
                    return {
                        "success": True,
                        **result
                    }
            return {
                "success": False,
                "error": "Knowledge graph not available"
            }
        except Exception as e:
            log.exception("Error in hybrid reclassification")
            return {
                "success": False,
                "error": str(e)
            }
    
    def reload_graph_ontology(self) -> Dict[str, Any]:
        """
        Reload ontology configuration from disk.
        
        This allows dynamic updates to node types, relationships, and
        keywords without restarting the application. After reloading,
        you may want to call reclassify_graph_nodes() to apply the
        new configuration to existing nodes.
        
        Returns:
            Dict with reload status and statistics
        """
        try:
            if hasattr(self.pdf_agent, 'pdf_parser') and hasattr(
                self.pdf_agent.pdf_parser, 'graph_manager'
            ):
                graph_mgr = self.pdf_agent.pdf_parser.graph_manager
                if graph_mgr is not None:
                    result = graph_mgr.reload_ontology()
                    return {
                        "success": True,
                        **result
                    }
            return {
                "success": False,
                "error": "Knowledge graph not available"
            }
        except Exception as e:
            log.exception("Error reloading ontology")
            return {
                "success": False,
                "error": str(e)
            }
    
    # ==================== Natural Language Workflows ====================
    
    def detect_paper_workflow_intent(self, query: str) -> Dict[str, Any]:
        """
        Detect if user wants to search/download/process papers from natural language
        
        Args:
            query: User's natural language query
        
        Returns:
            Dict with 'is_paper_workflow', 'actions', 'search_query', 
            'max_results', 'save_location'
        """
        import re
        
        query_lower = query.lower()
        
        # Keywords for paper search/download/process
        # Made more specific to avoid false positives with local PDF searches
        # "search the papers" = local search, "search for papers" = online search
        search_keywords = [
            'search for papers', 'find papers about', 'look for papers on',
            'search for academic papers', 'find academic papers',
            'search arxiv', 'search semantic scholar', 'search pubmed'
        ]
        download_keywords = [
            'download papers', 'get papers', 'fetch papers',
            'retrieve papers', 'download from'
        ]
        process_keywords = [
            'process papers', 'index papers', 'add papers to knowledge base', 
            'ingest papers', 'process downloaded'
        ]
        
        # Check if this is a paper workflow request
        has_search = any(keyword in query_lower for keyword in search_keywords)
        has_download = any(keyword in query_lower for keyword in download_keywords)
        has_process = any(keyword in query_lower for keyword in process_keywords)
        
        # Enhanced detection: Check for workflow patterns like "Search, download, and process papers"
        # This handles comma-separated action lists that don't use exact phrases
        workflow_pattern = r'\b(search|download|process)\b[,\s]+(and\s+)?\b(search|download|process)\b'
        if re.search(workflow_pattern, query_lower) and 'papers' in query_lower:
            # Found workflow pattern with "papers" keyword - enable individual actions
            if 'search' in query_lower:
                has_search = True
            if 'download' in query_lower:
                has_download = True
            if 'process' in query_lower:
                has_process = True
        
        if not (has_search or has_download or has_process):
            return {
                "is_paper_workflow": False,
                "actions": [],
                "search_query": None,
                "max_results": None,
                "save_location": None
            }
        
        # Extract number of papers (e.g., "50 papers", "download 20 papers")
        max_results = None
        number_pattern = r'\b(\d+)\s+papers?\b'
        number_match = re.search(number_pattern, query_lower)
        if number_match:
            max_results = int(number_match.group(1))
        
        # Extract save location (e.g., "save in downloads/agent2agent", "save to /path/to/dir")
        save_location = None
        save_patterns = [
            r'save\s+(?:in|to|at)\s+([^\s.,;]+(?:/[^\s.,;]+)*)',
            r'save\s+(?:the\s+)?papers?\s+(?:in|to|at)\s+([^\s.,;]+(?:/[^\s.,;]+)*)',
            r'to\s+([^\s.,;]+(?:/[^\s.,;]+)*)\s*$'  # "...to downloads/folder" at end
        ]
        for pattern in save_patterns:
            save_match = re.search(pattern, query_lower)
            if save_match:
                save_location = save_match.group(1).strip()
                break
        
        # Extract search query (remove workflow keywords and connectors)
        # First, try to extract content after key prepositions
        extracted = None
        for prep in [' about ', ' on ', ' for ', ' regarding ', ' concerning ']:
            if prep in query_lower:
                extracted = query_lower.split(prep, 1)[-1].strip()
                break
        
        # If no preposition found, use the whole query
        if not extracted:
            extracted = query_lower
        
        # Remove save location from extracted query if present
        if save_location:
            for pattern in save_patterns:
                extracted = re.sub(pattern, '', extracted)
        
        # Remove number specification from query
        if max_results:
            extracted = re.sub(number_pattern, '', extracted)
        
        # Remove workflow-related words token by token
        remove_tokens = {
            'search', 'find', 'look', 'download', 'process', 'get', 'fetch',
            'and', 'or', 'the', 'a', 'an', 'papers', 'paper', 'them', 'it',
            'please', 'can', 'you', 'could', 'would', 'save', 'in', 'to', 'at',
            'per', 'source', 'sources'
        }
        
        # Split into words, filter out unwanted tokens, rejoin
        words = extracted.split()
        filtered_words = [w.strip(',.!?;:') for w in words if w.strip(',.!?;:') not in remove_tokens]
        search_query = ' '.join(filtered_words)
        
        # If query is too short or empty, fall back to original
        if not search_query or len(search_query) < 3:
            search_query = query.strip()
        
        # Determine workflow actions
        actions = []
        if has_search or has_download:
            actions.append('search')
        if has_download:
            actions.append('download')
        if has_process:
            actions.append('process')
        
        return {
            "is_paper_workflow": True,
            "actions": actions,
            "search_query": search_query,
            "max_results": max_results,
            "save_location": save_location
        }
    
    async def execute_paper_workflow(
        self, 
        search_query: str, 
        actions: List[str],
        max_results: int = 5,
        save_location: str = None,
        auto_process: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a complete paper search/download/process workflow
        
        Args:
            search_query: Query to search for papers
            actions: List of actions to perform ['search', 'download', 'process']
            max_results: Maximum number of papers to download
            save_location: Optional custom directory to save papers
            auto_process: Whether to automatically process without prompting
        
        Returns:
            Dict with workflow results
        """
        workflow_result = {
            "success": True,
            "search_query": search_query,
            "actions_completed": [],
            "search_results": None,
            "download_results": None,
            "process_results": None,
            "downloaded_files": [],
            "save_location": save_location,
            "error": None
        }
        
        try:
            # Step 1: Search (and optionally download)
            if 'download' in actions:
                log.info(f"Executing search and download workflow for: {search_query}")
                
                # Use custom save location if provided
                download_kwargs = {
                    "query": search_query,
                    "max_results": max_results
                }
                if save_location:
                    download_kwargs["custom_dir"] = save_location
                
                result = await self.search_and_download_papers(**download_kwargs)
                workflow_result["search_results"] = result
                workflow_result["download_results"] = result.get("downloads", {})
                workflow_result["actions_completed"].append("search")
                workflow_result["actions_completed"].append("download")
                
                # Collect downloaded files
                if result.get("downloads"):
                    for download in result["downloads"].get("results", []):
                        if download.get("success") and download.get("path"):
                            workflow_result["downloaded_files"].append(
                                Path(download["path"])
                            )
            
            elif 'search' in actions:
                log.info(f"Executing search workflow for: {search_query}")
                result = await self.search_papers(
                    query=search_query,
                    max_results_per_source=max_results,
                    download_pdfs=False
                )
                workflow_result["search_results"] = result
                workflow_result["actions_completed"].append("search")
            
            # Step 2: Process downloaded files
            if 'process' in actions and workflow_result["downloaded_files"]:
                log.info(f"Processing {len(workflow_result['downloaded_files'])} downloaded files")
                
                processed_count = 0
                failed_count = 0
                process_details = []
                
                for pdf_path in workflow_result["downloaded_files"]:
                    if pdf_path.exists():
                        result = self.process_pdf(pdf_path)
                        process_details.append(result)
                        if result["success"]:
                            processed_count += 1
                        else:
                            failed_count += 1
                
                workflow_result["process_results"] = {
                    "total_files": len(workflow_result["downloaded_files"]),
                    "processed": processed_count,
                    "failed": failed_count,
                    "details": process_details
                }
                workflow_result["actions_completed"].append("process")
                
            # Convert Path objects to strings for JSON serialization
            if workflow_result.get("downloaded_files"):
                workflow_result["downloaded_files"] = [str(p) for p in workflow_result["downloaded_files"]]
            
            return workflow_result
            
        except Exception as e:
            log.exception(f"Error in paper workflow: {search_query}")
            workflow_result["success"] = False
            workflow_result["error"] = str(e)
            return workflow_result
    
    # ==================== Paper Search ====================
    
    async def search_papers(self, query: str, max_results_per_source: int = 10, download_pdfs: bool = False) -> Dict[str, Any]:
        """
        Search for academic papers
        
        Args:
            query: Search query
            max_results_per_source: Maximum results per search engine
            download_pdfs: Whether to download PDFs automatically
        
        Returns:
            Dict with search results
        """
        try:
            service = await self._ensure_paper_service()
            result = await service.search(
                query=query,
                max_results_per_source=max_results_per_source,
                download_pdfs=download_pdfs
            )
            return {
                "success": True,
                **result
            }
        except Exception as e:
            log.exception(f"Error searching papers: {query}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "total_results": 0
            }
    
    async def search_and_download_papers(
        self,
        query: str,
        max_results: int = 5,
        custom_dir: str = None
    ) -> Dict[str, Any]:
        """
        Search and download academic papers
        
        Args:
            query: Search query
            max_results: Maximum number of papers to download
            custom_dir: Optional custom directory to save papers
        
        Returns:
            Dict with search and download results
        """
        try:
            service = await self._ensure_paper_service()
            
            # Build kwargs for service call
            download_kwargs = {
                "query": query,
                "max_results": max_results
            }
            if custom_dir:
                download_kwargs["custom_dir"] = custom_dir
            
            result = await service.search_and_download(**download_kwargs)
            return {
                "success": True,
                **result
            }
        except Exception as e:
            log.exception(f"Error searching/downloading papers: {query}")
            return {
                "success": False,
                "error": str(e),
                "total_results": 0
            }
    
    def get_paper_sources(self) -> List[str]:
        """Get list of enabled paper search sources"""
        try:
            # This method is synchronous, so we need to handle it differently
            if self._paper_service is None:
                from services.paper_search_service import PaperSearchService
                self._paper_service = PaperSearchService()
            return self._paper_service.get_enabled_sources()
        except Exception as e:
            log.exception("Error getting paper sources")
            return []
    
    def get_paper_stats(self) -> Dict[str, Any]:
        """Get paper search statistics"""
        try:
            # This method is synchronous, so we need to handle it differently
            if self._paper_service is None:
                from services.paper_search_service import PaperSearchService
                self._paper_service = PaperSearchService()
            return self._paper_service.get_stats()
        except Exception as e:
            log.exception("Error getting paper stats")
            return {
                "total_searches": 0,
                "total_results": 0,
                "total_downloads": 0,
                "successful_downloads": 0,
                "failed_downloads": 0
            }
    
    # ==================== Memory & Session Management ====================
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history"""
        try:
            return self.pdf_agent.get_conversation_history(limit=limit)
        except Exception as e:
            log.exception("Error getting conversation history")
            return []
    
    def clear_memory(self) -> Dict[str, Any]:
        """Clear conversation memory"""
        try:
            self.pdf_agent.clear_memory()
            return {
                "success": True,
                "message": "Memory cleared successfully"
            }
        except Exception as e:
            log.exception("Error clearing memory")
            return {
                "success": False,
                "error": str(e)
            }
    
    def start_session(self, session_name: Optional[str] = None) -> Dict[str, Any]:
        """Start a new conversation session"""
        try:
            self.pdf_agent.start_session(session_name)
            return {
                "success": True,
                "session_name": session_name or "New Session"
            }
        except Exception as e:
            log.exception("Error starting session")
            return {
                "success": False,
                "error": str(e)
            }
    
    def end_session(self) -> Dict[str, Any]:
        """End current session"""
        try:
            self.pdf_agent.end_session()
            return {
                "success": True,
                "message": "Session ended"
            }
        except Exception as e:
            log.exception("Error ending session")
            return {
                "success": False,
                "error": str(e)
            }
    
    # ==================== Statistics & Configuration ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        try:
            return self.pdf_agent.get_stats()
        except Exception as e:
            log.exception("Error getting stats")
            return {
                "user_id": self.user_id,
                "error": str(e)
            }
    
    def get_providers(self) -> Dict[str, str]:
        """Get current AI provider configuration"""
        return {
            "embedding": settings.EMBEDDING_PROVIDER,
            "llm": settings.LLM_PROVIDER
        }
    
    def set_embedding_provider(self, provider: str) -> Dict[str, Any]:
        """
        Set embedding provider
        
        Args:
            provider: 'azure' or 'ollama'
        
        Returns:
            Dict with success status
        """
        if provider.lower() in ["azure", "ollama"]:
            settings.EMBEDDING_PROVIDER = provider.lower()
            return {
                "success": True,
                "provider": provider.lower(),
                "message": "Embedding provider updated (restart required)"
            }
        return {
            "success": False,
            "error": f"Invalid embedding provider: {provider}. Use 'azure' or 'ollama'"
        }
    
    def set_llm_provider(self, provider: str) -> Dict[str, Any]:
        """
        Set LLM provider
        
        Args:
            provider: 'poe' or 'ollama'
        
        Returns:
            Dict with success status
        """
        if provider.lower() in ["poe", "ollama"]:
            settings.LLM_PROVIDER = provider.lower()
            return {
                "success": True,
                "provider": provider.lower(),
                "message": "LLM provider updated (restart required)"
            }
        return {
            "success": False,
            "error": f"Invalid LLM provider: {provider}. Use 'poe' or 'ollama'"
        }
    
    # ==================== Lifecycle ====================
    
    def shutdown(self):
        """Gracefully shutdown the agent"""
        log.info("Shutting down ResearchAssistantAgent...")
        self.pdf_agent.shutdown()
        log.info("ResearchAssistantAgent shutdown complete")
    
    def detect_literature_review_intent(self, query: str) -> Dict[str, Any]:
        """
        Detect if user wants to create a literature review from natural language
        
        Args:
            query: User's natural language query
        
        Returns:
            Dict with 'is_literature_review', 'title', 'topic', 'select_all'
        """
        import re
        
        query_lower = query.lower()
        
        # Keywords for literature review creation
        create_keywords = [
            'create a literature review', 'create literature review',
            'generate a literature review', 'generate literature review',
            'make a literature review', 'make literature review'
        ]
        
        if not any(keyword in query_lower for keyword in create_keywords):
            return {
                "is_literature_review": False,
                "title": None,
                "topic": None,
                "select_all": False
            }
            
        # Extract title
        title = "Literature Review" # Default
        title_match = re.search(r'(?:named|titled|with the name|with title)\s+["\']?([^"\']+?)["\']?(?:\s+selecting|\s+using|\s+with|\s*$)', query_lower)
        if title_match:
            title = title_match.group(1).strip()
            
        # Check for "select all" intent
        select_all = 'select all' in query_lower or 'selecting all' in query_lower or 'use all' in query_lower or 'using all' in query_lower
        
        # Extract topic filter
        topic = None
        topic_match = re.search(r'(?:about|on|regarding|covering)\s+(.+?)(?:\s*$)', query_lower)
        if topic_match:
            topic = topic_match.group(1).strip()
            
        return {
            "is_literature_review": True,
            "title": title,
            "topic": topic,
            "select_all": select_all
        }
    
    # ==================== Advanced Workflow Methods ====================
    
    async def execute_research_workflow(
        self,
        prompt: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Execute a complex multi-step research workflow from natural language.
        
        This handles sophisticated workflows like:
        - Doctoral dissertation planning
        - Literature reviews with topic selection
        - Multi-stage research projects
        
        Args:
            prompt: Natural language description of the research workflow
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict with workflow results, steps, and outputs
            
        Example:
            ```python
            result = await agent.execute_research_workflow(
                "Propose 5 topics on AI agents, download 10 papers per topic, "
                "select the best topic, and justify the selection"
            )
            ```
        """
        try:
            orchestrator = self._get_workflow_orchestrator()
            
            if progress_callback:
                orchestrator.register_progress_callback(progress_callback)
            
            result = await orchestrator.execute_workflow(prompt)
            
            return result
            
        except Exception as e:
            log.exception("Error executing research workflow")
            return {
                "success": False,
                "error": str(e)
            }
    
    def detect_research_workflow_intent(self, query: str) -> Dict[str, Any]:
        """
        Detect if query is a complex research workflow that requires orchestration.
        
        Detects patterns like:
        - Doctoral research planning
        - Multi-topic literature reviews
        - Topic proposal + selection workflows
        
        Args:
            query: Natural language query
            
        Returns:
            Dict with detection results and workflow type
        """
        query_lower = query.lower()
        
        # Doctoral research indicators
        doctoral_indicators = [
            "doctoral", "dissertation", "phd", "scholar-practitioner",
            "research interest", "research topic area"
        ]
        
        # Multi-step workflow indicators
        multistep_indicators = [
            ("propose", "topics"),
            ("download", "papers", "topic"),
            ("select", "defend"),
            ("critically", "assess")
        ]
        
        # Count matches
        doctoral_matches = sum(1 for ind in doctoral_indicators if ind in query_lower)
        multistep_matches = sum(
            1 for ind in multistep_indicators 
            if all(word in query_lower for word in ind)
        )
        
        is_research_workflow = doctoral_matches >= 2 or multistep_matches >= 2
        
        # Determine workflow type
        workflow_type = None
        if doctoral_matches >= 2:
            workflow_type = "doctoral_research_planning"
        elif "literature review" in query_lower:
            workflow_type = "literature_review"
        elif multistep_matches >= 2:
            workflow_type = "multi_step_research"
        
        return {
            "is_research_workflow": is_research_workflow,
            "workflow_type": workflow_type,
            "confidence": (doctoral_matches + multistep_matches) / 5.0,
            "indicators": {
                "doctoral": doctoral_matches,
                "multistep": multistep_matches
            }
        }
    
    def get_workflow_progress(self) -> Dict[str, Any]:
        """
        Get progress of current workflow execution.
        
        Returns:
            Dict with current workflow status and progress
        """
        if self._workflow_orchestrator is None:
            return {"status": "no_active_workflow"}
        
        return self._workflow_orchestrator.get_progress()
    
    async def execute_workflow_step(
        self,
        step_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single workflow step manually.
        
        Useful for custom workflows or debugging.
        
        Args:
            step_type: Type of step (analysis, search, download, etc.)
            parameters: Step parameters
            
        Returns:
            Step execution result
        """
        try:
            from core.workflow_orchestrator import WorkflowStep, StepType, StepStatus
            
            orchestrator = self._get_workflow_orchestrator()
            
            step = WorkflowStep(
                id="manual_step",
                type=StepType(step_type),
                description=parameters.get("description", "Manual step"),
                parameters=parameters
            )
            
            await orchestrator._execute_step(step)
            
            return {
                "success": step.status == StepStatus.COMPLETED,
                "result": step.result,
                "error": step.error
            }
            
        except Exception as e:
            log.exception("Error executing manual workflow step")
            return {
                "success": False,
                "error": str(e)
            }
