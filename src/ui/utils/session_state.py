"""
Session State Management

This module provides utilities for managing session state in web applications,
particularly useful for Streamlit apps or other stateful web interfaces.
"""

import json
import pickle
import threading
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict


class SessionStateManager:
    """
    Thread-safe session state manager for web applications.
    
    Supports in-memory storage with optional persistence to disk.
    """
    
    def __init__(self, persist_to_disk: bool = False, storage_path: Optional[str] = None):
        """
        Initialize session state manager.
        
        Args:
            persist_to_disk: Whether to persist session data to disk
            storage_path: Path for persistent storage (default: ./sessions/)
        """
        self._storage = defaultdict(dict)
        self._metadata = defaultdict(dict)
        self._lock = threading.RLock()
        self._persist_to_disk = persist_to_disk
        self._storage_path = Path(storage_path or "./sessions/")
        
        if self._persist_to_disk:
            self._storage_path.mkdir(exist_ok=True)
            self._load_from_disk()
    
    def get(self, session_id: str, key: str, default: Any = None) -> Any:
        """
        Get value from session state.
        
        Args:
            session_id: Session identifier
            key: Key to retrieve
            default: Default value if key not found
            
        Returns:
            Value from session state or default
        """
        with self._lock:
            return self._storage[session_id].get(key, default)
    
    def set(self, session_id: str, key: str, value: Any) -> None:
        """
        Set value in session state.
        
        Args:
            session_id: Session identifier
            key: Key to set
            value: Value to store
        """
        with self._lock:
            self._storage[session_id][key] = value
            self._metadata[session_id]["last_updated"] = datetime.now()
            
            if self._persist_to_disk:
                self._save_session_to_disk(session_id)
    
    def update(self, session_id: str, key: str, update_func: Callable[[Any], Any]) -> Any:
        """
        Update value in session state using a function.
        
        Args:
            session_id: Session identifier
            key: Key to update
            update_func: Function to apply to current value
            
        Returns:
            Updated value
        """
        with self._lock:
            current_value = self._storage[session_id].get(key)
            new_value = update_func(current_value)
            self._storage[session_id][key] = new_value
            self._metadata[session_id]["last_updated"] = datetime.now()
            
            if self._persist_to_disk:
                self._save_session_to_disk(session_id)
            
            return new_value
    
    def delete(self, session_id: str, key: str) -> bool:
        """
        Delete key from session state.
        
        Args:
            session_id: Session identifier
            key: Key to delete
            
        Returns:
            True if key was deleted, False if not found
        """
        with self._lock:
            if key in self._storage[session_id]:
                del self._storage[session_id][key]
                self._metadata[session_id]["last_updated"] = datetime.now()
                
                if self._persist_to_disk:
                    self._save_session_to_disk(session_id)
                
                return True
            return False
    
    def clear_session(self, session_id: str) -> None:
        """
        Clear all data for a session.
        
        Args:
            session_id: Session identifier
        """
        with self._lock:
            self._storage[session_id].clear()
            self._metadata[session_id].clear()
            
            if self._persist_to_disk:
                session_file = self._storage_path / f"{session_id}.json"
                if session_file.exists():
                    session_file.unlink()
    
    def session_exists(self, session_id: str) -> bool:
        """
        Check if session exists.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session exists
        """
        with self._lock:
            return bool(self._storage[session_id])
    
    def get_all_keys(self, session_id: str) -> List[str]:
        """
        Get all keys for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of keys in session
        """
        with self._lock:
            return list(self._storage[session_id].keys())
    
    def get_session_data(self, session_id: str) -> Dict[str, Any]:
        """
        Get all data for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary of session data
        """
        with self._lock:
            return dict(self._storage[session_id])
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up old sessions.
        
        Args:
            max_age_hours: Maximum age of sessions in hours
            
        Returns:
            Number of sessions cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        with self._lock:
            sessions_to_remove = []
            
            for session_id, metadata in self._metadata.items():
                last_updated = metadata.get("last_updated")
                if last_updated and last_updated < cutoff_time:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                self.clear_session(session_id)
                cleaned_count += 1
        
        return cleaned_count
    
    def _save_session_to_disk(self, session_id: str) -> None:
        """Save session data to disk."""
        session_file = self._storage_path / f"{session_id}.json"
        try:
            with open(session_file, 'w') as f:
                data = {
                    'session_data': self._storage[session_id],
                    'metadata': self._metadata[session_id]
                }
                json.dump(data, f, default=str, indent=2)
        except Exception as e:
            print(f"Warning: Could not save session {session_id} to disk: {e}")
    
    def _load_from_disk(self) -> None:
        """Load session data from disk."""
        if not self._storage_path.exists():
            return
        
        for session_file in self._storage_path.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                
                session_id = session_file.stem
                self._storage[session_id] = data.get('session_data', {})
                self._metadata[session_id] = data.get('metadata', {})
                
            except Exception as e:
                print(f"Warning: Could not load session file {session_file}: {e}")


# Global session manager instance
_global_session_manager = SessionStateManager()


def get_session_value(session_id: str, key: str, default: Any = None) -> Any:
    """
    Get value from global session state.
    
    Args:
        session_id: Session identifier
        key: Key to retrieve
        default: Default value if key not found
        
    Returns:
        Value from session state or default
    """
    return _global_session_manager.get(session_id, key, default)


def set_session_value(session_id: str, key: str, value: Any) -> None:
    """
    Set value in global session state.
    
    Args:
        session_id: Session identifier
        key: Key to set
        value: Value to store
    """
    _global_session_manager.set(session_id, key, value)


def update_session_value(session_id: str, key: str, update_func: Callable[[Any], Any]) -> Any:
    """
    Update value in global session state using a function.
    
    Args:
        session_id: Session identifier
        key: Key to update
        update_func: Function to apply to current value
        
    Returns:
        Updated value
    """
    return _global_session_manager.update(session_id, key, update_func)


def clear_session(session_id: str) -> None:
    """
    Clear all data for a session in global session state.
    
    Args:
        session_id: Session identifier
    """
    _global_session_manager.clear_session(session_id)


def session_exists(session_id: str) -> bool:
    """
    Check if session exists in global session state.
    
    Args:
        session_id: Session identifier
        
    Returns:
        True if session exists
    """
    return _global_session_manager.session_exists(session_id)


def init_session_defaults(session_id: str, defaults: Dict[str, Any]) -> None:
    """
    Initialize session with default values if they don't exist.
    
    Args:
        session_id: Session identifier
        defaults: Dictionary of default values
    """
    for key, value in defaults.items():
        if get_session_value(session_id, key) is None:
            set_session_value(session_id, key, value)


def get_or_create_session_list(session_id: str, key: str) -> List[Any]:
    """
    Get a list from session state or create empty list if it doesn't exist.
    
    Args:
        session_id: Session identifier
        key: Key for the list
        
    Returns:
        List from session state
    """
    current_list = get_session_value(session_id, key, [])
    if not isinstance(current_list, list):
        current_list = []
        set_session_value(session_id, key, current_list)
    return current_list


def session_counter(session_id: str, key: str, increment: int = 1) -> int:
    """
    Increment a counter in session state.
    
    Args:
        session_id: Session identifier
        key: Key for the counter
        increment: Amount to increment (default: 1)
        
    Returns:
        New counter value
    """
    def increment_func(current_value):
        return (current_value or 0) + increment
    
    return update_session_value(session_id, key, increment_func)


# Streamlit-specific utilities
try:
    import streamlit as st
    
    def init_streamlit_session_defaults(defaults: Dict[str, Any]) -> None:
        """
        Initialize Streamlit session state with default values.
        
        Args:
            defaults: Dictionary of default values
        """
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def get_streamlit_session_value(key: str, default: Any = None) -> Any:
        """
        Get value from Streamlit session state with default.
        
        Args:
            key: Key to retrieve
            default: Default value if key not found
            
        Returns:
            Value from session state or default
        """
        return st.session_state.get(key, default)
    
    def toggle_streamlit_session_bool(key: str, default: bool = False) -> bool:
        """
        Toggle a boolean value in Streamlit session state.
        
        Args:
            key: Key for the boolean
            default: Default value if key doesn't exist
            
        Returns:
            New boolean value
        """
        current_value = st.session_state.get(key, default)
        st.session_state[key] = not current_value
        return st.session_state[key]

except ImportError:
    # Streamlit not available
    pass