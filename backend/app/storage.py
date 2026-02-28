"""
JSON file-based storage module
Replaces MongoDB with simple JSON file storage for local/open-source usage.
Each "collection" is a JSON file in the data/ directory.
"""
import json
import os
import uuid
import asyncio
import threading
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Data directory — writable at runtime (not inside the frozen bundle)
from app.paths import get_runtime_path
DATA_DIR = get_runtime_path("data")


def _json_serializer(obj):
    """Custom JSON serializer for datetime and other types"""
    import math
    import numpy as np
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    try:
        import numpy as np
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _parse_dates(obj):
    """Try to parse ISO format date strings back to datetime objects"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                try:
                    # Try parsing ISO format dates
                    if 'T' in value and len(value) > 18:
                        obj[key] = datetime.fromisoformat(value)
                except (ValueError, TypeError):
                    pass
            elif isinstance(value, dict):
                _parse_dates(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _parse_dates(item)
    return obj


def generate_id() -> str:
    """Generate a unique ID (replaces ObjectId)"""
    return uuid.uuid4().hex[:24]


class InsertResult:
    """Mimics MongoDB InsertOneResult"""
    def __init__(self, inserted_id: str):
        self.inserted_id = inserted_id


class JsonCursor:
    """
    Async-iterable cursor that mimics MongoDB cursor behavior.
    Supports .sort() chaining.
    """
    def __init__(self, docs: List[Dict]):
        self._docs = list(docs)  # copy
        self._index = 0
    
    def sort(self, key: str, direction: int = 1):
        """Sort documents by key. direction=1 for ascending, -1 for descending."""
        def sort_key(x):
            val = x.get(key, '')
            if val is None:
                return ''
            if isinstance(val, datetime):
                return val.isoformat()
            return val
        self._docs.sort(key=sort_key, reverse=(direction == -1))
        return self
    
    def __aiter__(self):
        self._index = 0
        return self
    
    async def __anext__(self):
        if self._index >= len(self._docs):
            raise StopAsyncIteration
        doc = self._docs[self._index]
        self._index += 1
        return doc


class JsonStore:
    """
    Async JSON file-based document store.
    Mimics MongoDB collection API for easy migration.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.file_path = DATA_DIR / f"{name}.json"
        self._lock = asyncio.Lock()
        self._file_lock = threading.Lock()  # Protects concurrent file reads/writes
        self._ensure_file()
    
    def _ensure_file(self):
        """Ensure the data directory and JSON file exist"""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            with open(self.file_path, 'w') as f:
                json.dump([], f)
    
    def _read_sync(self) -> List[Dict]:
        """Read all documents from JSON file (synchronous, thread-safe)"""
        try:
            with self._file_lock:
                with open(self.file_path, 'r') as f:
                    docs = json.load(f)
            return [_parse_dates(doc) for doc in docs]
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _write_sync(self, docs: List[Dict]):
        """Write all documents to JSON file (synchronous, thread-safe)"""
        with self._file_lock:
            with open(self.file_path, 'w') as f:
                json.dump(docs, f, indent=2, default=_json_serializer)
    
    def _matches(self, doc: Dict, query: Dict) -> bool:
        """Check if a document matches a query filter"""
        for key, value in query.items():
            # Handle $or operator
            if key == '$or':
                if isinstance(value, list) and value:
                    if not any(self._matches(doc, sub_q) for sub_q in value):
                        return False
                continue
            
            if key.startswith('$'):
                continue  # Skip other unsupported operators at top level
            
            # Handle $exists operator
            if isinstance(value, dict) and "$exists" in value:
                exists = value["$exists"]
                field_exists = key in doc
                if exists and not field_exists:
                    return False
                if not exists and field_exists:
                    return False
                continue
            
            # Handle regular value matching
            if doc.get(key) != value:
                return False
        return True
    
    async def find_one(self, query: Dict) -> Optional[Dict]:
        """Find a single document matching the query"""
        async with self._lock:
            docs = self._read_sync()
            for doc in docs:
                if self._matches(doc, query):
                    return doc
            return None
    
    def find(self, query: Dict = None) -> JsonCursor:
        """
        Find all documents matching query. Returns a JsonCursor.
        Note: This reads synchronously but returns an async-iterable cursor.
        """
        docs = self._read_sync()
        if query:
            matched_docs = [doc for doc in docs if self._matches(doc, query)]
            return JsonCursor(matched_docs)
        return JsonCursor(docs)
    
    async def insert_one(self, doc: Dict) -> InsertResult:
        """Insert a document. Auto-generates _id if not present."""
        async with self._lock:
            docs = self._read_sync()
            if "_id" not in doc:
                doc["_id"] = generate_id()
            doc_id = doc["_id"]
            docs.append(doc)
            self._write_sync(docs)
            return InsertResult(doc_id)
    
    async def update_one(self, query: Dict, update: Dict):
        """
        Update a single document matching the query.
        Supports MongoDB-style $set, $inc, and $unset operators, 
        or plain dict for direct field updates.
        """
        async with self._lock:
            docs = self._read_sync()
            for i, doc in enumerate(docs):
                if self._matches(doc, query):
                    if "$set" in update:
                        for key, value in update["$set"].items():
                            # Support nested keys like "stats.models_trained"
                            self._set_nested(doc, key, value)
                    if "$inc" in update:
                        for key, value in update["$inc"].items():
                            current = self._get_nested(doc, key, 0)
                            self._set_nested(doc, key, current + value)
                    if "$unset" in update:
                        for key in update["$unset"].keys():
                            # Remove the field from the document
                            if '.' in key:
                                # Handle nested keys
                                parts = key.split('.')
                                current = doc
                                for part in parts[:-1]:
                                    if isinstance(current, dict) and part in current:
                                        current = current[part]
                                    else:
                                        break
                                else:
                                    if isinstance(current, dict) and parts[-1] in current:
                                        del current[parts[-1]]
                            else:
                                # Simple key
                                if key in doc:
                                    del doc[key]
                    if "$set" not in update and "$inc" not in update and "$unset" not in update:
                        # Plain update (direct field assignment)
                        for key, value in update.items():
                            if not key.startswith('$'):
                                doc[key] = value
                    docs[i] = doc
                    self._write_sync(docs)
                    return
    
    async def delete_one(self, query: Dict):
        """Delete a single document matching the query"""
        async with self._lock:
            docs = self._read_sync()
            for i, doc in enumerate(docs):
                if self._matches(doc, query):
                    docs.pop(i)
                    self._write_sync(docs)
                    return
    
    async def count_documents(self, query: Dict = None) -> int:
        """Count documents matching query"""
        async with self._lock:
            docs = self._read_sync()
            if not query:
                return len(docs)
            return sum(1 for doc in docs if self._matches(doc, query))
    
    @staticmethod
    def _get_nested(doc: Dict, key: str, default=None):
        """Get a nested value from a dict using dot notation"""
        parts = key.split('.')
        current = doc
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current
    
    @staticmethod
    def _set_nested(doc: Dict, key: str, value):
        """Set a nested value in a dict using dot notation"""
        parts = key.split('.')
        current = doc
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value


# Store instances cache
_stores: Dict[str, JsonStore] = {}


def get_store(collection_name: str) -> JsonStore:
    """
    Get a JsonStore instance for a collection.
    Replaces get_collection() from the old MongoDB module.
    
    Args:
        collection_name: Name of the collection (datasets, models, automl_models)
    
    Returns:
        JsonStore instance
    """
    if collection_name not in _stores:
        _stores[collection_name] = JsonStore(collection_name)
    return _stores[collection_name]


def init_storage():
    """Initialize the storage system (create data directory)"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Storage initialized at {DATA_DIR}")
