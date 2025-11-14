"""
utils/caching_system.py

Multi-level caching system for RAG performance optimization.
Implements the caching strategies described in Week 3 blog post.

Provides L1 (memory), L2 (context), and L3 (persistent) caching layers
with LRU eviction, query normalization, and cost tracking.
"""

import json
import time
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict
from datetime import datetime, timedelta
import threading
import sqlite3

from utils.logger import log


class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, updating LRU order"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache, evicting LRU if necessary"""
        with self.lock:
            if key in self.cache:
                # Update existing key
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Evict least recently used
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        with self.lock:
            return len(self.cache)


class QueryNormalizer:
    """Normalize queries for consistent caching"""
    
    @staticmethod
    def normalize_query(query: str) -> str:
        """Normalize query string for cache key generation"""
        # Convert to lowercase
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove common punctuation that doesn't affect meaning
        normalized = normalized.replace('?', '').replace('!', '').replace('.', '')
        
        # Handle common synonyms and variations
        replacements = {
            'what are': 'what is',
            'how do': 'how does',
            'can you': '',
            'please': '',
            'tell me about': 'about',
            'explain': 'what is',
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        # Remove extra spaces after replacements
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    @staticmethod
    def generate_cache_key(query: str, mode: str, session_id: str = "default", 
                          additional_params: Optional[Dict] = None) -> str:
        """Generate consistent cache key"""
        normalized_query = QueryNormalizer.normalize_query(query)
        
        key_components = [normalized_query, mode, session_id]
        
        if additional_params:
            # Sort parameters for consistency
            sorted_params = sorted(additional_params.items())
            params_str = json.dumps(sorted_params, sort_keys=True)
            key_components.append(params_str)
        
        cache_key = "::".join(key_components)
        
        # Generate hash for consistent key length
        return hashlib.md5(cache_key.encode()).hexdigest()


class PersistentCache:
    """SQLite-based persistent cache for expensive operations"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    timestamp REAL,
                    access_count INTEGER DEFAULT 1,
                    size_bytes INTEGER
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON cache_entries(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_access_count ON cache_entries(access_count)
            """)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from persistent cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT value, access_count FROM cache_entries WHERE key = ?", 
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    value_blob, access_count = row
                    
                    # Update access count and timestamp
                    cursor.execute("""
                        UPDATE cache_entries 
                        SET access_count = ?, timestamp = ?
                        WHERE key = ?
                    """, (access_count + 1, time.time(), key))
                    
                    # Deserialize value
                    return pickle.loads(value_blob)
            
            return None
            
        except Exception as e:
            log.warning(f"Error reading from persistent cache: {e}")
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in persistent cache"""
        try:
            # Serialize value
            value_blob = pickle.dumps(value)
            size_bytes = len(value_blob)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, value, timestamp, size_bytes)
                    VALUES (?, ?, ?, ?)
                """, (key, value_blob, time.time(), size_bytes))
            
        except Exception as e:
            log.warning(f"Error writing to persistent cache: {e}")
    
    def cleanup_old_entries(self, max_age_hours: int = 168):  # 1 week default
        """Remove old cache entries"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM cache_entries WHERE timestamp < ?", 
                    (cutoff_time,)
                )
                deleted_count = cursor.rowcount
                
                if deleted_count > 0:
                    log.info(f"Cleaned up {deleted_count} old cache entries")
                    
        except Exception as e:
            log.warning(f"Error cleaning up persistent cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get basic stats
                cursor.execute("SELECT COUNT(*), SUM(size_bytes) FROM cache_entries")
                count, total_size = cursor.fetchone()
                
                # Get most accessed entries
                cursor.execute("""
                    SELECT key, access_count 
                    FROM cache_entries 
                    ORDER BY access_count DESC 
                    LIMIT 5
                """)
                top_entries = cursor.fetchall()
                
                return {
                    'entry_count': count or 0,
                    'total_size_bytes': total_size or 0,
                    'total_size_mb': (total_size or 0) / (1024 * 1024),
                    'top_entries': top_entries
                }
                
        except Exception as e:
            log.warning(f"Error getting cache stats: {e}")
            return {'error': str(e)}


class CostTracker:
    """Track API costs for cloud providers"""
    
    def __init__(self):
        self.total_tokens = 0
        self.total_requests = 0
        self.provider_costs = {
            'azure_openai': {
                'embedding_cost_per_1k': 0.0001,  # text-embedding-ada-002
                'completion_cost_per_1k': 0.002   # gpt-3.5-turbo
            },
            'poe': {
                'cost_per_message': 0.01  # Estimated
            },
            'ollama': {
                'cost_per_token': 0.0  # Local, no API costs
            }
        }
        self.usage_log = []
    
    def add_usage(self, provider: str, operation: str, 
                  tokens: int = 0, requests: int = 1):
        """Record API usage"""
        self.total_tokens += tokens
        self.total_requests += requests
        
        usage_entry = {
            'timestamp': time.time(),
            'provider': provider,
            'operation': operation,
            'tokens': tokens,
            'requests': requests
        }
        
        self.usage_log.append(usage_entry)
        
        # Keep only recent entries (last 1000)
        if len(self.usage_log) > 1000:
            self.usage_log = self.usage_log[-1000:]
    
    def get_estimated_cost(self, provider: str) -> float:
        """Calculate estimated cost for a provider"""
        if provider not in self.provider_costs:
            return 0.0
        
        provider_usage = [
            entry for entry in self.usage_log 
            if entry['provider'] == provider
        ]
        
        total_tokens = sum(entry['tokens'] for entry in provider_usage)
        total_requests = sum(entry['requests'] for entry in provider_usage)
        
        costs = self.provider_costs[provider]
        
        if provider == 'azure_openai':
            return (total_tokens / 1000) * costs['embedding_cost_per_1k']
        elif provider == 'poe':
            return total_requests * costs['cost_per_message']
        else:  # ollama
            return 0.0
    
    def get_total_cost(self) -> float:
        """Get total estimated cost across all providers"""
        return sum(
            self.get_estimated_cost(provider) 
            for provider in self.provider_costs.keys()
        )


class RAGCachingSystem:
    """Multi-level caching system for RAG performance optimization"""
    
    def __init__(self, cache_dir: str = "storage/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # L1: In-memory query cache (recent queries)
        self.query_cache = LRUCache(max_size=50)
        
        # L2: Context cache (retrieved contexts by query signature)
        self.context_cache = LRUCache(max_size=200)
        
        # L3: Persistent embedding cache (expensive embeddings)
        self.persistent_cache = PersistentCache(
            self.cache_dir / "persistent_cache.db"
        )
        
        # Cost tracking
        self.cost_tracker = CostTracker()
        
        # Cache statistics
        self.stats = {
            'query_hits': 0,
            'query_misses': 0,
            'context_hits': 0,
            'context_misses': 0,
            'persistent_hits': 0,
            'persistent_misses': 0
        }
        
        log.info(f"RAG caching system initialized at {cache_dir}")
    
    def get_cached_response(self, query: str, mode: str, 
                           session_id: str = "default",
                           max_age_seconds: int = 3600) -> Optional[Dict]:
        """Check for cached complete response (L1 cache)"""
        
        cache_key = QueryNormalizer.generate_cache_key(query, mode, session_id)
        
        cached_result = self.query_cache.get(cache_key)
        
        if cached_result:
            # Check if cache is still fresh
            if time.time() - cached_result['timestamp'] < max_age_seconds:
                self.stats['query_hits'] += 1
                log.debug(f"L1 Cache HIT: {cache_key[:8]}...")
                return cached_result
            else:
                # Remove stale cache entry
                self.query_cache.put(cache_key, None)
        
        self.stats['query_misses'] += 1
        return None
    
    def cache_response(self, query: str, mode: str, session_id: str,
                      response: str, context: str, metadata: Dict):
        """Cache complete response for future use (L1 cache)"""
        
        cache_key = QueryNormalizer.generate_cache_key(query, mode, session_id)
        
        cached_result = {
            'query': query,
            'mode': mode,
            'session_id': session_id,
            'response': response,
            'context': context,
            'metadata': metadata,
            'timestamp': time.time()
        }
        
        self.query_cache.put(cache_key, cached_result)
        log.debug(f"L1 Cache STORE: {cache_key[:8]}...")
    
    def get_cached_context(self, query_signature: str) -> Optional[List]:
        """Get cached retrieval context (L2 cache)"""
        
        cached_context = self.context_cache.get(query_signature)
        
        if cached_context:
            self.stats['context_hits'] += 1
            log.debug(f"L2 Cache HIT: Context for {query_signature[:8]}...")
            return cached_context
        
        self.stats['context_misses'] += 1
        return None
    
    def cache_context(self, query_signature: str, context_nodes: List):
        """Cache retrieval context for reuse (L2 cache)"""
        
        self.context_cache.put(query_signature, context_nodes)
        log.debug(f"L2 Cache STORE: Context for {query_signature[:8]}...")
    
    def get_cached_embedding(self, text_hash: str) -> Optional[List[float]]:
        """Get cached embedding (L3 persistent cache)"""
        
        embedding = self.persistent_cache.get(f"embedding:{text_hash}")
        
        if embedding:
            self.stats['persistent_hits'] += 1
            log.debug(f"L3 Cache HIT: Embedding for {text_hash[:8]}...")
            return embedding
        
        self.stats['persistent_misses'] += 1
        return None
    
    def cache_embedding(self, text: str, embedding: List[float], 
                       provider: str = "unknown"):
        """Cache expensive embedding (L3 persistent cache)"""
        
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        cache_entry = {
            'embedding': embedding,
            'provider': provider,
            'text_length': len(text),
            'timestamp': time.time()
        }
        
        self.persistent_cache.put(f"embedding:{text_hash}", cache_entry)
        log.debug(f"L3 Cache STORE: Embedding for {text_hash[:8]}...")
        
        # Track embedding generation cost
        estimated_tokens = len(text) // 4  # Rough approximation
        self.cost_tracker.add_usage(provider, 'embedding', estimated_tokens)
    
    def generate_query_signature(self, query: str, mode: str, 
                                additional_params: Optional[Dict] = None) -> str:
        """Generate normalized query signature for context caching"""
        
        return QueryNormalizer.generate_cache_key(
            query, mode, "context", additional_params
        )
    
    def cleanup_caches(self, max_age_hours: int = 24):
        """Clean up old cache entries"""
        
        log.info("Cleaning up cache entries...")
        
        # Clean persistent cache
        self.persistent_cache.cleanup_old_entries(max_age_hours)
        
        # Clear in-memory caches (they're session-based anyway)
        if max_age_hours < 1:  # Only if very aggressive cleanup
            self.query_cache.clear()
            self.context_cache.clear()
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        
        query_total = self.stats['query_hits'] + self.stats['query_misses']
        context_total = self.stats['context_hits'] + self.stats['context_misses']
        persistent_total = self.stats['persistent_hits'] + self.stats['persistent_misses']
        
        persistent_stats = self.persistent_cache.get_stats()
        
        return {
            'query_cache': {
                'hits': self.stats['query_hits'],
                'misses': self.stats['query_misses'],
                'hit_rate': self.stats['query_hits'] / query_total if query_total > 0 else 0,
                'size': self.query_cache.size()
            },
            'context_cache': {
                'hits': self.stats['context_hits'],
                'misses': self.stats['context_misses'],
                'hit_rate': self.stats['context_hits'] / context_total if context_total > 0 else 0,
                'size': self.context_cache.size()
            },
            'persistent_cache': {
                'hits': self.stats['persistent_hits'],
                'misses': self.stats['persistent_misses'],
                'hit_rate': self.stats['persistent_hits'] / persistent_total if persistent_total > 0 else 0,
                **persistent_stats
            },
            'cost_tracking': {
                'total_tokens': self.cost_tracker.total_tokens,
                'total_requests': self.cost_tracker.total_requests,
                'estimated_total_cost': self.cost_tracker.get_total_cost(),
                'cost_by_provider': {
                    provider: self.cost_tracker.get_estimated_cost(provider)
                    for provider in self.cost_tracker.provider_costs.keys()
                }
            }
        }
    
    def warm_cache(self, common_queries: List[str], mode: str = "enhanced"):
        """Pre-populate cache with common queries"""
        
        log.info(f"Warming cache with {len(common_queries)} common queries...")
        
        # This would typically involve running the queries through the system
        # For now, we'll just generate the cache keys to reserve space
        for query in common_queries:
            cache_key = QueryNormalizer.generate_cache_key(query, mode)
            # Cache structure is ready for when these queries are actually processed
            log.debug(f"Cache key prepared for: {query[:30]}...")


class RAGErrorHandler:
    """Error handling with caching integration"""
    
    def __init__(self, cache_system: RAGCachingSystem, 
                 fallback_provider: str = "ollama"):
        self.cache_system = cache_system
        self.fallback_provider = fallback_provider
        self.error_count = defaultdict(int)
        self.max_retries = 3
        self.backoff_base = 2
    
    def safe_cached_operation(self, operation_func, *args, 
                             cache_key: Optional[str] = None, 
                             **kwargs):
        """Execute operation with caching and error handling"""
        
        # Try cache first if cache_key provided
        if cache_key:
            cached_result = self.cache_system.persistent_cache.get(cache_key)
            if cached_result:
                return cached_result
        
        # Execute operation with retry logic
        for attempt in range(self.max_retries):
            try:
                result = operation_func(*args, **kwargs)
                
                # Cache successful result
                if cache_key and result is not None:
                    self.cache_system.persistent_cache.put(cache_key, result)
                
                return result
                
            except Exception as e:
                self.error_count['operation_failures'] += 1
                log.warning(f"Operation attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    delay = self.backoff_base ** attempt
                    time.sleep(delay)
        
        # All attempts failed
        log.error(f"Operation failed after {self.max_retries} attempts")
        return None


# Global cache instance
_global_cache = None


def get_cache_system(cache_dir: str = "storage/cache") -> RAGCachingSystem:
    """Get or create global cache system instance"""
    global _global_cache
    
    if _global_cache is None:
        _global_cache = RAGCachingSystem(cache_dir)
    
    return _global_cache