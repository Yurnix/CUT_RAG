import chromadb
from chromadb.config import Settings
from typing import List, Optional, Dict, Any
import uuid

class ChromaManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB manager with optional persistence directory.
        
        Args:
            persist_directory (str): Directory where ChromaDB will store its data
        """
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            is_persistent=True
        ))
        self.collection = self.create_collection()

    def create_collection(self, collection_name: str = "documents") -> chromadb.Collection:
        """
        Create a new collection or get existing one.
        
        Args:
            collection_name (str): Name of the collection
            
        Returns:
            chromadb.Collection: The created or retrieved collection
        """
        try:
            # Try to get existing collection
            return self.client.get_collection(collection_name)
        except chromadb.errors.InvalidCollectionException:
            # Create new collection if it doesn't exist
            return self.client.create_collection(collection_name)
    
    def add_document(self, 
                    document: str, 
                    metadata: Optional[Dict[str, Any]] = None, 
                    doc_id: Optional[str] = None) -> str:
        """
        Add a document to the ChromaDB collection.
        
        Args:
            document (str): The document text to add
            metadata (Optional[Dict[str, Any]]): Optional metadata for the document
            doc_id (Optional[str]): Optional document ID. If not provided, a UUID will be generated
            
        Returns:
            str: The ID of the added document
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())
            
        self.collection.add(
            documents=[document],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )
        
        return doc_id
    
    def query_similar(self, 
                     query_text: str, 
                     n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query the collection for documents similar to the input text.
        
        Args:
            query_text (str): The text to find similar documents for
            n_results (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with their metadata and distances
        """
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
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
            
        return formatted_results
    
    def flush(self) -> None:
        """
        Delete all documents from the collection.
        """
        # Get all document IDs
        all_ids = self.collection.get()["ids"]
        if all_ids:
            # Delete all documents
            self.collection.delete(ids=all_ids)

    def get_collection_stats(self) -> Dict[str, int]:
        """
        Get basic statistics about the collection.
        
        Returns:
            Dict[str, int]: Dictionary containing collection statistics
        """
        count = self.collection.count()
        return {
            "total_documents": count
        }
