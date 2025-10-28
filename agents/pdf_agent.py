# agents/pdf_agent.py
from typing import Optional, Dict, List
from pathlib import Path

from core.pdf_parser import PDFParser
from core.memory_manager import MemoryManager
from core.context_manager import ContextManager
from core.search_engine import SearchEngine
from utils.file_watcher import FileWatcher
from utils.logger import log
from config import settings


class PDFAgent:
    """Main AI Agent for PDF parsing, search, and context management"""
    
    def __init__(self, user_id: str = "default", auto_watch: Optional[bool] = None):
        log.info("Initializing PDFAgent...")
        
        self.user_id = user_id
        
        # Initialize components
        self.pdf_parser = PDFParser()
        self.memory_manager = MemoryManager(user_id=user_id)
        
        # Load or create index
        log.info("Loading vector index...")
        self.index = self.pdf_parser.load_existing_index()
        
        if self.index is None:
            log.info("No existing index found, will create on first document processing")
        
        # Initialize context and search
        self.context_manager = None
        self.search_engine = None
        
        if self.index is not None:
            graph_mgr = getattr(self.pdf_parser, 'graph_manager', None)
            self.context_manager = ContextManager(
                self.index, 
                self.memory_manager,
                graph_manager=graph_mgr
            )
            self.search_engine = SearchEngine(
                self.context_manager, 
                self.memory_manager
            )
        
        # File watcher
        self.file_watcher = None
        if auto_watch is None:
            auto_watch = settings.AUTO_WATCH
        
        if auto_watch:
            self.setup_file_watcher()
        
        log.info("PDFAgent initialized successfully!")
    
    def setup_file_watcher(self):
        """Setup automatic file watching"""
        self.file_watcher = FileWatcher(
            callback=self._handle_new_file,
            watch_path=settings.DOWNLOADS_FOLDER
        )
        self.file_watcher.start()
    
    def _handle_new_file(self, file_path: Path):
        """Callback for new file detection"""
        log.info(f"Processing new file: {file_path.name}")
        self.process_pdf(file_path)
    
    def _ensure_search_ready(self):
        """Ensure search components are initialized"""
        if self.index is None:
            raise RuntimeError("No index available. Please process some PDFs first.")
        
        if self.context_manager is None:
            graph_mgr = getattr(self.pdf_parser, 'graph_manager', None)
            self.context_manager = ContextManager(
                self.index, 
                self.memory_manager,
                graph_manager=graph_mgr
            )
        
        if self.search_engine is None:
            self.search_engine = SearchEngine(
                self.context_manager, 
                self.memory_manager
            )
    
    def process_pdf(self, file_path: Path) -> bool:
        """Process a single PDF file"""
        try:
            # Parse PDF
            documents = self.pdf_parser.parse_pdf(file_path)
            
            if not documents:
                log.warning(f"No documents extracted from {file_path.name}")
                return False
            
            # Process documents
            self.index = self.pdf_parser.process_documents(documents)
            
            # Reinitialize search components with new index
            graph_mgr = getattr(self.pdf_parser, 'graph_manager', None)
            self.context_manager = ContextManager(
                self.index, 
                self.memory_manager,
                graph_manager=graph_mgr
            )
            self.search_engine = SearchEngine(
                self.context_manager, 
                self.memory_manager
            )
            
            log.info(f"Successfully processed: {file_path.name}")
            return True
            
        except Exception as e:
            log.error(f"Failed to process {file_path}: {e}")
            return False
    
    def process_folder(self, folder_path: Optional[Path] = None) -> Dict:
        """Process all PDFs in a folder"""
        try:
            # Parse folder
            documents = self.pdf_parser.parse_folder(folder_path)
            
            if not documents:
                log.warning("No documents to process")
                return {"success": False, "message": "No documents found"}
            
            # Process documents
            self.index = self.pdf_parser.process_documents(documents)
            
            # Reinitialize search components
            graph_mgr = getattr(self.pdf_parser, 'graph_manager', None)
            self.context_manager = ContextManager(
                self.index, 
                self.memory_manager,
                graph_manager=graph_mgr
            )
            self.search_engine = SearchEngine(
                self.context_manager, 
                self.memory_manager
            )
            
            stats = self.get_stats()
            log.info(f"Folder processing complete: {stats}")
            
            return {
                "success": True,
                "documents_processed": len(documents),
                "stats": stats
            }
            
        except Exception as e:
            log.error(f"Failed to process folder: {e}")
            return {"success": False, "error": str(e)}
    
    def search(
        self, 
        query: str, 
        mode: str = "enhanced",
        save_to_memory: bool = True
    ) -> Dict:
        """Search the knowledge base"""
        self._ensure_search_ready()
        return self.search_engine.search(query, mode=mode, save_to_memory=save_to_memory)
    
    def chat(self, message: str, mode: Optional[str] = None) -> str:
        """Interactive chat with context awareness"""
        self._ensure_search_ready()
        
        # Detect if user wants comprehensive analysis of all papers
        analyze_all_keywords = [
            "analyze all", "all papers", "all documents", "every paper",
            "comprehensive review", "analyze each", "all the papers",
            "analyze the papers", "each paper", "all research"
        ]
        
        message_lower = message.lower()
        should_analyze_all = any(keyword in message_lower for keyword in analyze_all_keywords)
        
        # Choose mode based on query intent or provided mode
        if mode is None:
            mode = "analyze_all" if should_analyze_all else "enhanced"
        else:
            # Use provided mode, but log if it conflicts with detection
            detected_mode = "analyze_all" if should_analyze_all else "enhanced"
            if mode != detected_mode:
                log.info(f"Mode override: {mode} vs detected {detected_mode}")
        
        if should_analyze_all or mode == "analyze_all":
            log.info("Detected comprehensive analysis request")
        
        result = self.search_engine.search(message, mode=mode)
        return result["answer"]
    
    def get_sources(self, query: str) -> List[Dict]:
        """Get source documents for a query"""
        self._ensure_search_ready()
        return self.context_manager.get_context_sources(query)
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        return self.memory_manager.get_messages(limit=limit)
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory_manager.clear_memory()
        log.info("Conversation memory cleared")
    
    def start_session(self, session_name: Optional[str] = None):
        """Start a new conversation session"""
        self.memory_manager.start_session(session_name)
    
    def end_session(self):
        """End current session"""
        self.memory_manager.end_session()
    
    def get_stats(self) -> Dict:
        """Get agent statistics"""
        stats = {
            "user_id": self.user_id,
            "index_stats": self.pdf_parser.get_index_stats(),
            "memory_stats": self.memory_manager.get_memory_summary(),
            "watch_active": self.file_watcher is not None
        }
        return stats
    
    def shutdown(self):
        """Gracefully shutdown the agent"""
        log.info("Shutting down PDFAgent...")
        
        if self.file_watcher:
            self.file_watcher.stop()
        
        self.memory_manager.save()
        
        log.info("PDFAgent shutdown complete")