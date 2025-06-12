#!/usr/bin/env python3
"""
ChromaDB Handler - Handles all ChromaDB operations for video summaries using langchain_chroma
"""

from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper for sentence_transformers to use with LangChain"""
    
    def __init__(self, sentence_transformer):
        """Initialize with a sentence_transformer model"""
        self.sentence_transformer = sentence_transformer
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using the sentence transformer"""
        print(f"Embedding {len(texts)} documents")
        return self.sentence_transformer.encode(texts).tolist()
        
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the sentence transformer"""
        print(f"Embedding query: {text}")
        return self.sentence_transformer.encode(text).tolist()

class ChromaDBHandler:
    def __init__(self, embeddings, persist_directory="chroma_db"):
        """Initialize ChromaDB client with LangChain integration
        
        Args:
            embeddings: SentenceTransformer or embeddings function to use
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = persist_directory
        
        # Create embeddings wrapper if it's not already a LangChain Embeddings object
        if not isinstance(embeddings, Embeddings):
            print("Using SentenceTransformer for embeddings")
            self.embeddings = SentenceTransformerEmbeddings(embeddings)
        else:
            print("Using provided embeddings object")
            self.embeddings = embeddings
    
    def create_collection(self, collection_name: str, metadata: Optional[Dict] = None):
        """Create a new collection"""
        try:
            # LangChain Chroma will create a new collection or get existing one
            collection = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Delete all existing documents if the collection exists
            try:
                collection.delete_collection()
                collection = Chroma(
                    collection_name=collection_name, 
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )
            except:
                pass
                
            return collection
        except Exception as e:
            print(f"Error creating collection: {e}")
            return None
    
    def get_collection(self, collection_name: str):
        """Get existing collection"""
        try:
            return Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        except Exception as e:
            print(f"Collection '{collection_name}' not found: {e}")
            return None
    
    def get_or_create_collection(self, collection_name: str, metadata: Optional[Dict] = None):
        """Get existing collection or create new one"""
        collection = self.get_collection(collection_name)
        print(f"Got collection: {collection}")
        if collection is None:
            collection = self.create_collection(collection_name, metadata)
            print(f"Created collection: {collection}")
        return collection
    
    def add_documents(self, collection, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Add documents to collection"""
        try:
            # Convert to LangChain Document format
            langchain_docs = [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(documents, metadatas)
            ]
            
            # Add documents to collection
            collection.add_documents(
                documents=langchain_docs,
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
            
            results = collection.similarity_search_with_score(
                query=query,
                k=n_results
            )
            
            # Format results to match original format
            formatted_results = {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": []
            }
            
            for doc, score in results:
                formatted_results["documents"].append(doc.page_content)
                formatted_results["metadatas"].append(doc.metadata)
                formatted_results["ids"].append(doc.metadata.get("id", ""))
                formatted_results["distances"].append(score)
                
            return formatted_results
        except Exception as e:
            print(f"Error searching: {e}")
            return None
    
    def get_collection_info(self, collection_name: str):
        """Get collection information"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return None
            
            # Access the underlying Chroma collection
            chroma_collection = collection._collection
            count = chroma_collection.count()
            
            return {
                "name": collection_name,
                "count": count,
                "metadata": {}  # LangChain doesn't expose collection metadata directly
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return None
    
    def delete_collection(self, collection_name: str):
        """Delete collection"""
        try:
            collection = self.get_collection(collection_name)
            if collection:
                collection.delete_collection()
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
    
    def list_collections(self):
        """List all collections"""
        try:
            # Create temporary client to list collections
            from chromadb import PersistentClient
            client = PersistentClient(path=self.persist_directory)
            return client.list_collections()
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []