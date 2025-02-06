from typing import List, Optional, Dict, Any
import os
from pypdf import PdfReader
import pandas as pd
from chroma_manager import ChromaManager

class EmbeddingManager:
    def __init__(self, chroma_manager: Optional[ChromaManager] = None, symbol_threshold: int = 512):
        """
        Initialize EmbeddingManager with optional ChromaManager instance and symbol threshold.
        
        Args:
            chroma_manager (Optional[ChromaManager]): ChromaManager instance. If None, creates a new one
            symbol_threshold (int): Maximum number of symbols per chunk
        """
        self.chroma_manager = chroma_manager or ChromaManager()
        self.symbol_threshold = symbol_threshold
    
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
                    text += page.extract_text() + "\n"
            return text
            
        elif file_extension == '.csv':
            df = pd.read_csv(filepath)
            # Convert DataFrame to string representation
            return df.to_string()
            
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on symbol threshold.
        
        Args:
            text (str): Text to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        chunks = []
        current_chunk = ""
        
        # Split text into sentences (simple approach)
        sentences = text.replace('\n', ' ').split('.')
        
        for sentence in sentences:
            sentence = sentence.strip() + '.'  # Add back the period
            
            # If current chunk plus new sentence exceeds threshold, save current chunk
            if len(current_chunk) + len(sentence) > self.symbol_threshold:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += ' ' + sentence if current_chunk else sentence
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def add_file(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Process a file and add its chunks to ChromaDB.
        
        Args:
            filepath (str): Path to the file
            metadata (Optional[Dict[str, Any]]): Optional metadata for the chunks
            
        Returns:
            List[str]: List of document IDs for the added chunks
        """
        # Parse the file
        text = self.parse_file(filepath)
        
        # Chunk the text
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
            chunk_metadata = {**base_metadata, 'chunk_index': i}
            
            # Add to ChromaDB
            doc_id = self.chroma_manager.add_document(
                document=chunk,
                metadata=chunk_metadata
            )
            doc_ids.append(doc_id)
        
        return doc_ids
