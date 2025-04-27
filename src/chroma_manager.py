import chromadb
from chromadb.config import Settings
from typing import List, Optional, Dict, Any
import uuid
from interfaces import IEmbeddingManager

class ChromaManager(IEmbeddingManager):
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB manager with optional persistence directory.
        
        Args:
            persist_directory (str): Directory where ChromaDB will store its data
        """
        # Initialize collections cache
        self._collections_cache = {}
        
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            is_persistent=True
        ))
        self.default_collection_name = "documents"
        self.collection = self.create_collection()

    def create_collection(self, collection_name: str = None) -> chromadb.Collection:
        """
        Create a new collection or get existing one.
        
        Args:
            collection_name (str): Name of the collection
            
        Returns:
            chromadb.Collection: The created or retrieved collection
        """
        if collection_name is None:
            collection_name = self.default_collection_name
            
        if collection_name in self._collections_cache:
            return self._collections_cache[collection_name]
            
        try:
            # Try to get existing collection
            collection = self.client.get_collection(collection_name)
        except chromadb.errors.InvalidCollectionException:
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(collection_name)
            
        self._collections_cache[collection_name] = collection
        return collection
    
    def add_document(self, 
                    document: str, 
                    metadata: Optional[Dict[str, Any]] = None, 
                    doc_id: Optional[str] = None,
                    collection_name: Optional[str] = None) -> str:
        """
        Add a document to the ChromaDB collection.
        
        Args:
            document (str): The document text to add
            metadata (Optional[Dict[str, Any]]): Optional metadata for the document
            doc_id (Optional[str]): Optional document ID. If not provided, a UUID will be generated
            collection_name (Optional[str]): Name of the collection to add the document to
            
        Returns:
            str: The ID of the added document
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())
            
        collection = self.collection if collection_name is None else self.create_collection(collection_name)
            
        collection.add(
            documents=[document],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )
        
        return doc_id
    
    def query_similar(self, 
                     query_text: str, 
                     n_results: int = 5,
                     collection_names: Optional[List[str]] = None,
                     results_per_collection: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query the collection(s) for documents similar to the input text.
        
        Args:
            query_text (str): The text to find similar documents for
            n_results (int): Number of results to return
            collection_names (Optional[List[str]]): List of collections to query from. If None, uses default collection.
            results_per_collection (Optional[int]): Number of results to fetch from each collection.
                                                    If None, uses n_results for the default collection
                                                    or distributes evenly among specified collections.
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with their metadata and distances
        """
        # If no specific collections are given, query only the default collection
        if not collection_names:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            # Format results into a more user-friendly structure
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                    'collection': self.default_collection_name
                })
                
            return formatted_results
        
        # Query from multiple collections
        all_results = []
        
        # Determine how many results to get from each collection
        if results_per_collection is None:
            results_per_collection = max(1, n_results // len(collection_names))
        
        for collection_name in collection_names:
            collection = self.create_collection(collection_name)
            
            # Skip empty collections to avoid errors
            if collection.count() == 0:
                continue
                
            try:
                results = collection.query(
                    query_texts=[query_text],
                    n_results=results_per_collection
                )
                
                # Format and add collection results
                for i in range(len(results['ids'][0])):
                    all_results.append({
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None,
                        'collection': collection_name
                    })
            except Exception as e:
                print(f"Error querying collection {collection_name}: {str(e)}")
        
        # Sort by distance and limit to n_results
        all_results.sort(key=lambda x: x['distance'])
        return all_results[:n_results]
    
    def delete_documents_by_metadata(self, metadata_key: str, metadata_value: Any, collection_name: Optional[str] = None) -> None:
        """
        Delete all documents that match the given metadata key-value pair.
        
        Args:
            metadata_key (str): Metadata key to match
            metadata_value (Any): Value to match for the given key
            collection_name (Optional[str]): Name of the collection to delete from. If None, uses default collection.
        """
        collection = self.collection if collection_name is None else self.create_collection(collection_name)
        
        # Get all documents with matching metadata
        results = collection.get(
            where={metadata_key: metadata_value}
        )
        
        if results["ids"]:
            collection.delete(ids=results["ids"])
            
    def get_documents_by_metadata(self, metadata_key: str, metadata_value: Any) -> List[Dict[str, Any]]:
        """
        Get all documents that match the given metadata key-value pair.
        
        Args:
            metadata_key (str): Metadata key to match
            metadata_value (Any): Value to match for the given key
            
        Returns:
            List[Dict[str, Any]]: List of matching documents with their metadata
        """
        results = self.collection.get(
            where={metadata_key: metadata_value}
        )
        
        formatted_results = []
        for i in range(len(results["ids"])):
            formatted_results.append({
                'id': results["ids"][i],
                'document': results["documents"][i],
                'metadata': results["metadatas"][i]
            })
            
        return formatted_results
    
    def flush(self, collection_name: Optional[str] = None) -> None:
        """
        Delete all documents from the collection.
        
        Args:
            collection_name (Optional[str]): Name of the collection to flush. If None, flushes default collection.
        """
        if collection_name is None:
            # Get all document IDs from default collection
            all_ids = self.collection.get()["ids"]
            if all_ids:
                # Delete all documents
                self.collection.delete(ids=all_ids)
        else:
            # Flush specific collection
            collection = self.create_collection(collection_name)
            all_ids = collection.get()["ids"]
            if all_ids:
                collection.delete(ids=all_ids)

    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, int]:
        """
        Get basic statistics about the collection.
        
        Args:
            collection_name (Optional[str]): Name of the collection to get stats for. If None, uses default collection.
            
        Returns:
            Dict[str, int]: Dictionary containing collection statistics
        """
        if collection_name is None:
            count = self.collection.count()
            return {
                "total_documents": count
            }
        else:
            collection = self.create_collection(collection_name)
            count = collection.count()
            return {
                "total_documents": count
            }
            
    def list_collections(self) -> List[str]:
        """
        List all collections in the database.
        
        Returns:
            List[str]: List of collection names
        """
        return self.client.list_collections()
