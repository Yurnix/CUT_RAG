import re
import hashlib
from typing import List
from pypdf import PdfReader
from interfaces import IChunker, ChunkMetadata, TextChunk

class PdfChunker(IChunker):
    def __init__(self):
        pass

    def _create_text_hash(self, text: str) -> str:
        """Create a hash for the text content."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def chunk_document(self, file_path: str) -> List[TextChunk]:
        """
        Parse PDF and return chunks with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of TextChunk objects containing the text and metadata
        """
        chunks = []
        reader = PdfReader(file_path)
        file_name = file_path.split('/')[-1]
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if not text.strip():
                continue
            
                    
            # Create hash for the paragraph
            text_hash = self._create_text_hash(text)
                
            # Create metadata
            metadata = ChunkMetadata(
                file_name=file_name,
                page_number=page_num,
                text_hash=text_hash,
            )
                
            # Create and append the chunk
            text += "Source: " + file_name + "\nPage: " + str(page_num) + "\n\n"
            chunk = TextChunk(text=text, metadata=metadata)
            chunks.append(chunk)
        
        return chunks
