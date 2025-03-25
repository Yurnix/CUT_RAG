import hashlib
from typing import List
from interfaces import IChunker, ChunkMetadata, TextChunk
import os

class TextChunker(IChunker):
    """Chunker implementation for text files."""
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
    
    def _create_text_hash(self, text: str) -> str:
        """Create a hash for the text content."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def chunk_document(self, file_path: str) -> List[TextChunk]:
        """
        Read and chunk a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of TextChunk objects containing the text and metadata
        """
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        if not text.strip():
            return []
            
        # Get file name
        file_name = os.path.basename(file_path)
        
        # Create chunks
        chunks = []
        start = 0
        chunk_number = 1
        
        while start < len(text):
            # Get chunk text
            chunk_text = text[start:start + self.chunk_size]
            if not chunk_text.strip():
                break
                
            # Create hash for the chunk
            text_hash = self._create_text_hash(chunk_text)
            
            # Create metadata
            metadata = ChunkMetadata(
                file_name=file_name,
                page_number=chunk_number,
                text_hash=text_hash
            )
            
            # Create and append chunk
            chunk = TextChunk(text=chunk_text, metadata=metadata)
            chunks.append(chunk)
            
            # Move to next chunk
            start += self.chunk_size
            chunk_number += 1
        
        return chunks
