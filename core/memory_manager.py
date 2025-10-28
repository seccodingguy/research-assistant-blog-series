# core/memory_manager.py
from llama_index.core.memory import ChatMemoryBuffer, VectorMemory
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage
from typing import List, Dict, Optional
from pathlib import Path
import json
from datetime import datetime

from config import settings
from utils.logger import log


class MemoryManager:
    """Manage conversation memory and context"""
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.chat_store_path = settings.CHAT_HISTORY_PATH / f"{user_id}_chat_store.json"
        
        # Initialize chat store
        if self.chat_store_path.exists():
            self.chat_store = SimpleChatStore.from_persist_path(
                str(self.chat_store_path)
            )
            log.info(f"Loaded existing chat store for user: {user_id}")
        else:
            self.chat_store = SimpleChatStore()
            log.info(f"Created new chat store for user: {user_id}")
        
        # Initialize memory buffer
        self.memory = ChatMemoryBuffer.from_defaults(
            token_limit=settings.MAX_MEMORY_TOKENS,
            chat_store=self.chat_store,
            chat_store_key=user_id
        )
        
        # Conversation metadata
        self.metadata_path = settings.CHAT_HISTORY_PATH / f"{user_id}_metadata.json"
        self.metadata = self._load_metadata()
        
        log.info(f"MemoryManager initialized for user: {user_id}")
    
    def _load_metadata(self) -> Dict:
        """Load conversation metadata"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {
            "user_id": self.user_id,
            "created_at": datetime.now().isoformat(),
            "total_messages": 0,
            "sessions": []
        }
    
    def _save_metadata(self):
        """Save conversation metadata"""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to memory"""
        # Create ChatMessage object with additional_kwargs for metadata
        message = ChatMessage(
            role=role,
            content=content,
            additional_kwargs={
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
        )
        
        self.memory.put(message)
        self.metadata["total_messages"] += 1
        
        log.debug(f"Added {role} message to memory")
    
    def get_messages(self, limit: Optional[int] = None) -> List[Dict]:
        """Get conversation messages"""
        messages = self.memory.get()
        if limit:
            messages = messages[-limit:]
        return messages
    
    def get_conversation_context(self) -> str:
        """Get formatted conversation context"""
        messages = self.get_messages()
        context_parts = []
        
        for msg in messages:
            # Handle both ChatMessage objects and dictionaries for compatibility
            if isinstance(msg, ChatMessage):
                role = msg.role
                content = msg.content
            else:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            context_parts.append(f"{role.upper()}: {content}")
        
        return "\n".join(context_parts)
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.reset()
        log.info("Memory cleared")
    
    def start_session(self, session_name: Optional[str] = None):
        """Start a new conversation session"""
        session = {
            "session_id": len(self.metadata["sessions"]) + 1,
            "name": session_name or f"Session {len(self.metadata['sessions']) + 1}",
            "started_at": datetime.now().isoformat(),
            "message_count": 0
        }
        self.metadata["sessions"].append(session)
        self._save_metadata()
        log.info(f"Started new session: {session['name']}")
    
    def end_session(self):
        """End current session"""
        if self.metadata["sessions"]:
            current_session = self.metadata["sessions"][-1]
            current_session["ended_at"] = datetime.now().isoformat()
            current_session["message_count"] = self.metadata["total_messages"]
            self._save_metadata()
            log.info(f"Ended session: {current_session['name']}")
    
    def save(self):
        """Persist memory to disk"""
        self.chat_store.persist(str(self.chat_store_path))
        self._save_metadata()
        log.info("Memory saved to disk")
    
    def get_memory_summary(self) -> Dict:
        """Get summary of memory state"""
        messages = self.get_messages()
        
        # Calculate approximate token count
        total_tokens = 0
        for msg in messages:
            # Handle both ChatMessage objects and dictionaries for compatibility
            if isinstance(msg, ChatMessage):
                content = msg.content
            else:
                content = msg.get("content", "")
            # Rough estimation: ~4 characters per token
            total_tokens += len(content) // 4
        
        return {
            "user_id": self.user_id,
            "total_messages": len(messages),
            "sessions": len(self.metadata["sessions"]),
            "current_token_count": total_tokens,
            "token_limit": settings.MAX_MEMORY_TOKENS
        }