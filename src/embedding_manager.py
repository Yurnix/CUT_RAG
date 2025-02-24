from typing import List, Optional, Dict, Any
import os
from pypdf import PdfReader
import pandas as pd
from chroma_manager import ChromaManager
from chunker import Chunker, ChunkingMethod

class EmbeddingManager:
    def __init__(self, chroma_manager: Optional[ChromaManager] = None, 
                 chunk_size: int = 512, chunking_method: ChunkingMethod = ChunkingMethod.FIXED_LENGTH):
        """
        Initialize EmbeddingManager with optional ChromaManager instance and chunking parameters.
        
        Args:
            chroma_manager (Optional[ChromaManager]): ChromaManager instance. If None, creates a new one
            chunk_size (int): Maximum size of each chunk in tokens
            chunking_method (ChunkingMethod): Method to use for text chunking
        """
        self.chroma_manager = chroma_manager or ChromaManager()
        self.chunk_size = chunk_size
        self.chunking_method = chunking_method
        self.chunker = Chunker()
    
    def parse_file(self, filepath: str) -> str:
        """
        Parse different file formats and return their content as text.
        
        Args:
            filepath (str): Path to the file
            
        Returns:
            str: Extracted text content
        
        Raises:
            ValueError: If file format is not supported
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        file_extension = os.path.splitext(filepath)[1].lower()
        
        if file_extension == '.txt':
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
                
        elif file_extension == '.pdf':
            text = ""
            with open(filepath, 'rb') as file:
                pdf = PdfReader(file)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
            
        elif file_extension == '.csv':
            df = pd.read_csv(filepath)
            # Convert DataFrame to string representation
            return df.to_string()
            
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using the specified chunking method.
        
        Args:
            text (str): Text to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        return self.chunker.chunk_text(text, method=self.chunking_method, chunk_size=self.chunk_size)
    
    def flush_db(self):
        """Flush all data from the ChromaDB."""
        self.chroma_manager.flush()
        
    def add_file(self, filepath: str, metadata: Optional[Dict[str, Any]] = None, text_content: Optional[str] = None) -> List[str]:
        """
        Process a file and add its chunks to ChromaDB.
        
        Args:
            filepath (str): Path to the file
            metadata (Optional[Dict[str, Any]]): Optional metadata for the chunks
            text_content (Optional[str]): Optional pre-processed text content. If provided, skips file parsing
            
        Returns:
            List[str]: List of document IDs for the added chunks
        """
        if text_content is not None:
            # Use provided text content directly
            chunks = [text_content]  # Don't chunk pre-processed content
        else:
            # Parse the file and chunk it
            text = self.parse_file(filepath)
            chunks = self.chunk_text(text)
        
        # Add base metadata
        base_metadata = {
            'source_file': os.path.basename(filepath),
            'file_type': os.path.splitext(filepath)[1].lower()[1:],  # Remove the dot
        }
        if metadata:
            base_metadata.update(metadata)
        
        # Add chunks to ChromaDB
        doc_ids = []
        for i, chunk in enumerate(chunks):
            # Add chunk index to metadata
            chunk_metadata = {**base_metadata}
            
            # Add to ChromaDB
            doc_id = self.chroma_manager.add_document(
                document=chunk,
                metadata=chunk_metadata
            )
            doc_ids.append(doc_id)
        
        return doc_ids
