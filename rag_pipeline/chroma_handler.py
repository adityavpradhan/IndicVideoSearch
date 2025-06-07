#!/usr/bin/env python3
"""
ChromaDB Handler - Handles all ChromaDB operations for video summaries
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional

class ChromaDBHandler:
    def __init__(self, persist_directory="chroma_db"):
        """Initialize ChromaDB client"""
        self.persist_directory = persist_directory
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            is_persistent=True
        ))
    
    def create_collection(self, collection_name: str, metadata: Optional[Dict] = None):
        """Create a new collection"""
        try:
            # Delete existing collection if it exists
            try:
                self.client.delete_collection(collection_name)
            except:
                pass
            
            collection = self.client.create_collection(
                name=collection_name,
                metadata=metadata or {"description": "Video chunk summaries with temporal information"}
            )
            return collection
        except Exception as e:
            print(f"Error creating collection: {e}")
            return None
    
    def get_collection(self, collection_name: str):
        """Get existing collection"""
        try:
            return self.client.get_collection(collection_name)
        except Exception as e:
            print(f"Collection '{collection_name}' not found: {e}")
            return None
    
    def get_or_create_collection(self, collection_name: str, metadata: Optional[Dict] = None):
        """Get existing collection or create new one"""
        collection = self.get_collection(collection_name)
        if collection is None:
            collection = self.create_collection(collection_name, metadata)
        return collection
    
    def add_documents(self, collection, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Add documents to collection"""
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            return True
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False
    
    def search(self, collection_name: str, query: str, n_results: int = 5):
        """Search in collection"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return None
            
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"Error searching: {e}")
            return None
    
    def get_collection_info(self, collection_name: str):
        """Get collection information"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return None
            
            count = collection.count()
            return {
                "name": collection_name,
                "count": count,
                "metadata": collection.metadata
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return None
    
    def delete_collection(self, collection_name: str):
        """Delete collection"""
        try:
            self.client.delete_collection(collection_name)
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
    
    def list_collections(self):
        """List all collections"""
        try:
            return self.client.list_collections()
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []