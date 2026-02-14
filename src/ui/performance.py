# -*- coding: utf-8 -*-
"""
Phase 4 - ITEM-008: Performance Optimization

Features:
- Response caching for AI calls (avoid duplicate expensive requests)
- Project data lazy loading
- Async operation batching
- Memory-efficient file handling
- Progress indicators for long operations
- Debounced UI updates
"""

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from functools import wraps
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    LRU cache for expensive operations (AI calls, file I/O)
    
    Features:
    - Time-based expiration
    - Size limits (max entries)
    - Cache hit/miss metrics
    - Persistent cache (optional)
    """
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600, persist_path: Optional[Path] = None):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.persist_path = persist_path
        self.cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, timestamp)
        self.hits = 0
        self.misses = 0
        
        if persist_path and persist_path.exists():
            self._load_from_disk()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from function arguments"""
        key_data = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if exists and not expired"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            age = time.time() - timestamp
            
            if age < self.ttl_seconds:
                self.hits += 1
                logger.debug(f"Cache HIT: {key} (age: {age:.1f}s)")
                return value
            else:
                # Expired
                del self.cache[key]
                logger.debug(f"Cache EXPIRED: {key} (age: {age:.1f}s)")
        
        self.misses += 1
        logger.debug(f"Cache MISS: {key}")
        return None
    
    def set(self, key: str, value: Any):
        """Store value in cache with current timestamp"""
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
            logger.debug(f"Cache EVICT: {oldest_key}")
        
        self.cache[key] = (value, time.time())
        logger.debug(f"Cache SET: {key}")
        
        if self.persist_path:
            self._save_to_disk()
    
    def clear(self):
        """Clear all cached entries"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'ttl_seconds': self.ttl_seconds
        }
    
    def _save_to_disk(self):
        """Persist cache to disk"""
        try:
            if not self.persist_path:
                return
            
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Only save recent entries (within TTL)
            now = time.time()
            valid_cache = {
                k: v for k, v in self.cache.items()
                if (now - v[1]) < self.ttl_seconds
            }
            
            with open(self.persist_path, 'w') as f:
                json.dump(valid_cache, f)
            
            logger.debug(f"Cache saved to {self.persist_path} ({len(valid_cache)} entries)")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _load_from_disk(self):
        """Load cache from disk"""
        try:
            if not self.persist_path or not self.persist_path.exists():
                return
            
            with open(self.persist_path, 'r') as f:
                loaded = json.load(f)
            
            # Filter out expired entries
            now = time.time()
            self.cache = {
                k: tuple(v) for k, v in loaded.items()
                if (now - v[1]) < self.ttl_seconds
            }
            
            logger.info(f"Cache loaded from {self.persist_path} ({len(self.cache)} entries)")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")


def cached(cache: ResponseCache, key_func: Optional[Callable] = None):
    """
    Decorator to cache function results
    
    Usage:
        cache = ResponseCache()
        
        @cached(cache)
        async def expensive_ai_call(prompt: str) -> str:
            return await deepseek_client.call(...)
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache._generate_key(*args, **kwargs)
            
            # Check cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Call function and cache result
            result = await func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache._generate_key(*args, **kwargs)
            
            # Check cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class LazyLoader:
    """
    Lazy load project data to avoid loading all projects at startup
    
    Features:
    - Load project metadata on demand
    - Cache loaded projects
    - Background preloading for better UX
    """
    
    def __init__(self, projects_dir: Path):
        self.projects_dir = projects_dir
        self.loaded_projects: Dict[str, Dict] = {}
        self.metadata_cache: Dict[str, Dict] = {}  # Light metadata (name, type, created)
    
    def list_project_metadata(self) -> List[Dict[str, Any]]:
        """
        Get lightweight project metadata (name, type, created) without loading full config
        
        Returns list of dicts with: name, project_type, created_at, path
        """
        if self.metadata_cache:
            return list(self.metadata_cache.values())
        
        metadata_list = []
        
        if not self.projects_dir.exists():
            return metadata_list
        
        for project_path in self.projects_dir.iterdir():
            if not project_path.is_dir():
                continue
            
            config_file = project_path / 'config.json'
            if not config_file.exists():
                continue
            
            try:
                # Load only essential fields
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                metadata = {
                    'name': config.get('name', project_path.name),
                    'project_type': config.get('metadata', {}).get('project_type', 'api_dev'),
                    'domain': config.get('domain', ''),
                    'framework': config.get('framework', ''),
                    'created_at': config.get('created_at', ''),
                    'path': str(project_path),
                    'project_id': config.get('project_id', project_path.name)
                }
                
                self.metadata_cache[metadata['name']] = metadata
                metadata_list.append(metadata)
            
            except Exception as e:
                logger.warning(f"Failed to load metadata for {project_path.name}: {e}")
                continue
        
        return metadata_list
    
    def load_project(self, project_name: str) -> Optional[Dict]:
        """
        Fully load project config (called on-demand when project is selected)
        """
        if project_name in self.loaded_projects:
            return self.loaded_projects[project_name]
        
        if project_name not in self.metadata_cache:
            self.list_project_metadata()  # Refresh cache
        
        if project_name not in self.metadata_cache:
            logger.warning(f"Project {project_name} not found")
            return None
        
        project_path = Path(self.metadata_cache[project_name]['path'])
        config_file = project_path / 'config.json'
        
        try:
            with open(config_file, 'r') as f:
                full_config = json.load(f)
            
            self.loaded_projects[project_name] = full_config
            logger.info(f"Loaded full config for project: {project_name}")
            return full_config
        
        except Exception as e:
            logger.error(f"Failed to load project {project_name}: {e}")
            return None
    
    def invalidate_cache(self, project_name: Optional[str] = None):
        """Invalidate cache for specific project or all projects"""
        if project_name:
            self.loaded_projects.pop(project_name, None)
            self.metadata_cache.pop(project_name, None)
        else:
            self.loaded_projects.clear()
            self.metadata_cache.clear()
        
        logger.info(f"Cache invalidated for: {project_name or 'all projects'}")


class Debouncer:
    """
    Debounce UI updates to avoid excessive redraws
    
    Usage:
        debouncer = Debouncer(delay_ms=300)
        
        def on_text_change():
            debouncer.debounce(update_validation)
    """
    
    def __init__(self, delay_ms: int = 300):
        self.delay_ms = delay_ms
        self.pending_call: Optional[str] = None
    
    def debounce(self, func: Callable, after_id_var: List):
        """
        Debounce a function call
        
        Args:
            func: Function to call after delay
            after_id_var: List containing Tkinter after() ID (mutable reference)
        """
        # Cancel pending call
        if after_id_var and after_id_var[0]:
            try:
                func.__self__.after_cancel(after_id_var[0])
            except:
                pass
        
        # Schedule new call
        if hasattr(func, '__self__'):
            after_id = func.__self__.after(self.delay_ms, func)
            if after_id_var is not None:
                after_id_var[0] = after_id


class ProgressTracker:
    """
    Track progress for long operations with ETA calculation
    
    Features:
    - Progress percentage
    - ETA calculation based on throughput
    - Status messages
    """
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_times: List[float] = []
    
    def update(self, step: int, status: str = ""):
        """Update progress"""
        self.current_step = step
        self.step_times.append(time.time())
    
    def get_progress(self) -> float:
        """Get progress percentage (0-100)"""
        if self.total_steps == 0:
            return 0
        return (self.current_step / self.total_steps) * 100
    
    def get_eta(self) -> Optional[timedelta]:
        """Get estimated time remaining"""
        if self.current_step == 0 or len(self.step_times) < 2:
            return None
        
        elapsed = time.time() - self.start_time
        steps_per_second = self.current_step / elapsed
        
        if steps_per_second == 0:
            return None
        
        remaining_steps = self.total_steps - self.current_step
        eta_seconds = remaining_steps / steps_per_second
        
        return timedelta(seconds=int(eta_seconds))
    
    def reset(self):
        """Reset tracker"""
        self.current_step = 0
        self.start_time = time.time()
        self.step_times.clear()


class AsyncBatcher:
    """
    Batch async operations for better throughput
    
    Example:
        batcher = AsyncBatcher(batch_size=5, flush_interval=1.0)
        
        for item in items:
            await batcher.add(process_item, item)
        
        await batcher.flush()
    """
    
    def __init__(self, batch_size: int = 10, flush_interval: float = 1.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.pending: List[Tuple[Callable, Tuple, Dict]] = []
        self.last_flush = time.time()
    
    async def add(self, func: Callable, *args, **kwargs):
        """Add operation to batch"""
        self.pending.append((func, args, kwargs))
        
        # Auto-flush if batch full or interval elapsed
        if len(self.pending) >= self.batch_size or \
           (time.time() - self.last_flush) > self.flush_interval:
            return await self.flush()
    
    async def flush(self) -> List[Any]:
        """Execute all pending operations concurrently"""
        if not self.pending:
            return []
        
        tasks = [
            func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            for func, args, kwargs in self.pending
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        self.pending.clear()
        self.last_flush = time.time()
        
        return results


class MemoryOptimizer:
    """
    Memory optimization utilities
    
    Features:
    - Chunked file reading for large files
    - Generator-based processing
    - Memory usage monitoring
    """
    
    @staticmethod
    def read_large_file(file_path: Path, chunk_size: int = 8192):
        """
        Generator for reading large files in chunks
        
        Usage:
            for chunk in MemoryOptimizer.read_large_file(path):
                process(chunk)
        """
        with open(file_path, 'r') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    
    @staticmethod
    def process_batch(items: List[Any], batch_size: int = 100):
        """
        Generator for processing large lists in batches
        
        Usage:
            for batch in MemoryOptimizer.process_batch(large_list):
                results = process(batch)
        """
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]


# Global cache instance (optional - can be injected per-component)
_global_cache = ResponseCache(
    max_size=200,
    ttl_seconds=3600,  # 1 hour
    persist_path=Path.home() / '.ralph' / 'cache.json'
)


def get_global_cache() -> ResponseCache:
    """Get global cache instance"""
    return _global_cache
