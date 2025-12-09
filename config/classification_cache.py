# config/classification_cache.py
"""
Persistent cache for LLM-based node classifications.

This module provides caching to avoid re-classifying nodes with LLM,
reducing API costs and improving performance.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from utils.logger import log


class ClassificationCache:
    """Persistent cache for LLM classifications"""
    
    def __init__(self, cache_path: Optional[Path] = None):
        """
        Initialize classification cache.
        
        Args:
            cache_path: Path to cache file.
                       If None, uses config/llm_classification_cache.json
        """
        if cache_path is None:
            cache_path = Path(__file__).parent / "llm_classification_cache.json"
        
        self.cache_path = Path(cache_path)
        self._cache: Dict[str, Dict] = {}
        self._stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0
        }
        
        self.load()
    
    def load(self) -> None:
        """Load cache from disk"""
        try:
            if self.cache_path.exists():
                with open(self.cache_path, 'r') as f:
                    data = json.load(f)
                    self._cache = data.get('classifications', {})
                    self._stats = data.get('stats', self._stats)
                    log.info(
                        f"✓ Loaded classification cache "
                        f"({len(self._cache)} entries)"
                    )
            else:
                log.info("No existing classification cache found")
        except Exception as e:
            log.warning(f"Error loading classification cache: {e}")
            self._cache = {}
    
    def save(self) -> None:
        """Save cache to disk"""
        try:
            # Ensure directory exists
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'classifications': self._cache,
                'stats': self._stats,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            log.debug(f"Saved classification cache ({len(self._cache)} entries)")
            
        except Exception as e:
            log.error(f"Error saving classification cache: {e}")
    
    def get(self, label: str) -> Optional[str]:
        """
        Get cached classification for a label.
        
        Args:
            label: Node label
            
        Returns:
            Cached classification or None if not found
        """
        # Normalize label for consistent lookup
        key = label.lower().strip()
        
        if key in self._cache:
            self._stats['hits'] += 1
            entry = self._cache[key]
            log.debug(f"Cache hit for '{label}': {entry['classification']}")
            return entry['classification']
        else:
            self._stats['misses'] += 1
            return None
    
    def set(
        self,
        label: str,
        classification: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Cache a classification.
        
        Args:
            label: Node label
            classification: The classification result
            metadata: Optional metadata (context, confidence, etc.)
        """
        key = label.lower().strip()
        
        self._cache[key] = {
            'classification': classification,
            'original_label': label,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self._stats['saves'] += 1
        
        log.debug(f"Cached classification for '{label}': {classification}")
        
        # Auto-save every 10 new entries
        if self._stats['saves'] % 10 == 0:
            self.save()
    
    def clear(self) -> None:
        """Clear all cached classifications"""
        self._cache = {}
        self._stats = {'hits': 0, 'misses': 0, 'saves': 0}
        log.info("Cleared classification cache")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = (
            self._stats['hits'] / total_requests * 100
            if total_requests > 0
            else 0
        )
        
        return {
            'entries': len(self._cache),
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'saves': self._stats['saves'],
            'hit_rate': f"{hit_rate:.1f}%"
        }
    
    def export_classifications(self) -> Dict[str, str]:
        """
        Export all cached classifications as label -> type mapping.
        
        Returns:
            Dict mapping original labels to classifications
        """
        return {
            entry['original_label']: entry['classification']
            for entry in self._cache.values()
        }


# Global singleton instance
_classification_cache = None


def get_classification_cache(cache_path: Optional[Path] = None):
    """
    Get or create the global classification cache instance.
    
    Args:
        cache_path: Optional path to cache file (only used on first call)
        
    Returns:
        ClassificationCache instance
    """
    global _classification_cache
    
    if _classification_cache is None:
        _classification_cache = ClassificationCache(cache_path)
    
    return _classification_cache
