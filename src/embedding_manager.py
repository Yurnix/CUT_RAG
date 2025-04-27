from typing import List, Optional, Dict, Any
import os
import hashlib
from chroma_manager import ChromaManager
from pdf_chunker import PdfChunker
from text_chunker import TextChunker
from interfaces import IChunker, ChunkMetadata, TextChunk

class EmbeddingManager:
    def __init__(self, chroma_manager: Optional[ChromaManager] = None,
                 chunker: Optional[IChunker] = None):
        """
        Initialize EmbeddingManager with optional ChromaManager instance and chunking parameters.
        
        Args:
            chroma_manager (Optional[ChromaManager]): ChromaManager instance. If None, creates a new one
            chunker (Optional[IChunker]): Optional custom chunker implementation
        """
        self.chroma_manager = chroma_manager or ChromaManager()
        self.pdf_chunker = chunker or PdfChunker()
        self.text_chunker = TextChunker()
    
    def flush_db(self, collection_name: Optional[str] = None):
        """
        Flush data from the ChromaDB.
        
        Args:
            collection_name (Optional[str]): Name of the collection to flush. If None, flushes default collection.
        """
        self.chroma_manager.flush(collection_name)
    
    def get_topics(self) -> List[str]:
        """
        Get a list of available topics (collections) in the database,
        except the default collection.
        
        Returns:
            List[str]: List of topic names
        """
        collections = self.chroma_manager.list_collections()
        # Filter out the default collection
        return [c for c in collections if c != self.chroma_manager.default_collection_name]
        
    def add_file(self, filepath: str, metadata: Optional[Dict[str, Any]] = None, 
                text_content: Optional[str] = None, collection_name: Optional[str] = None) -> List[str]:
        """
        Process a file and add its chunks to ChromaDB.
        
        Args:
            filepath (str): Path to the file
            metadata (Optional[Dict[str, Any]]): Optional metadata for the chunks
            text_content (Optional[str]): Optional pre-processed text content. If provided, skips file parsing
            collection_name (Optional[str]): Name of the collection to add chunks to
            
        Returns:
            List[str]: List of document IDs for the added chunks
        """
        # Handle pre-processed text content
        if text_content is not None:
            chunks = [TextChunk(
                text=text_content,
                metadata=ChunkMetadata(
                    file_name=os.path.basename(filepath),
                    page_number=1,
                    text_hash=hashlib.sha256(text_content.encode()).hexdigest()[:16]
                )
            )]
        else:
            # Choose appropriate chunker based on file type
            file_extension = os.path.splitext(filepath)[1].lower()
            if file_extension == '.pdf':
                chunks = self.pdf_chunker.chunk_document(filepath)
            else:
                chunks = self.text_chunker.chunk_document(filepath)
        
        # Add base metadata
        base_metadata = {
            'source_file': os.path.basename(filepath),
            'file_type': os.path.splitext(filepath)[1].lower()[1:],  # Remove the dot
        }
        if metadata:
            base_metadata.update(metadata)
        
        # Add chunks to ChromaDB
        doc_ids = []
        for chunk in chunks:
            # Add chunk metadata
            chunk_metadata = {**base_metadata}
            chunk_metadata.update({
                'page_number': chunk.metadata.page_number,
                'text_hash': chunk.metadata.text_hash
            })
            
            # Add to ChromaDB with optional collection name
            doc_id = self.chroma_manager.add_document(
                document=chunk.text,
                metadata=chunk_metadata,
                collection_name=collection_name
            )
            doc_ids.append(doc_id)
        
        return doc_ids
