"""
Cache Manager - Simple caching utility with TTL support
"""

import time
import threading
from typing import Any, Dict, Optional, Union
from collections import OrderedDict


class CacheManager:
    """A thread-safe cache manager with TTL (Time To Live) support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of items in cache
            default_ttl: Default time to live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._ttls: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                return default
            
            # Check if item has expired
            if self._is_expired(key):
                self._remove_key(key)
                return default
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache with optional TTL."""
        with self._lock:
            if ttl is None:
                ttl = self.default_ttl
            
            # Remove existing key if it exists
            if key in self._cache:
                self._remove_key(key)
            
            # Add new item
            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._ttls[key] = ttl
            
            # Enforce max size
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                self._remove_key(oldest_key)
    
    def delete(self, key: str) -> bool:
        """Delete item from cache. Returns True if item existed."""
        with self._lock:
            if key in self._cache:
                self._remove_key(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all items from cache."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._ttls.clear()
    
    def has_key(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            if key not in self._cache:
                return False
            
            if self._is_expired(key):
                self._remove_key(key)
                return False
            
            return True
    
    def keys(self) -> list:
        """Get all non-expired keys."""
        with self._lock:
            self._cleanup_expired()
            return list(self._cache.keys())
    
    def size(self) -> int:
        """Get current cache size (excluding expired items)."""
        with self._lock:
            self._cleanup_expired()
            return len(self._cache)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            self._cleanup_expired()
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'default_ttl': self.default_ttl,
                'keys': list(self._cache.keys())
            }
    
    def _is_expired(self, key: str) -> bool:
        """Check if a key has expired."""
        if key not in self._timestamps:
            return True
        
        timestamp = self._timestamps[key]
        ttl = self._ttls.get(key, self.default_ttl)
        return time.time() - timestamp > ttl
    
    def _remove_key(self, key: str) -> None:
        """Remove key from all internal dictionaries."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
        self._ttls.pop(key, None)
    
    def _cleanup_expired(self) -> None:
        """Remove all expired items from cache."""
        expired_keys = [
            key for key in self._cache.keys()
            if self._is_expired(key)
        ]
        
        for key in expired_keys:
            self._remove_key(key)


# Global cache instance
_global_cache = CacheManager()

def get_cache() -> CacheManager:
    """Get the global cache instance."""
    return _global_cache

def cache_get(key: str, default: Any = None) -> Any:
    """Get item from global cache."""
    return _global_cache.get(key, default)

def cache_set(key: str, value: Any, ttl: Optional[int] = None) -> None:
    """Set item in global cache."""
    _global_cache.set(key, value, ttl)

def cache_delete(key: str) -> bool:
    """Delete item from global cache."""
    return _global_cache.delete(key)