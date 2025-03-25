from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ChunkMetadata:
    file_name: str
    page_number: int
    text_hash: str

@dataclass
class TextChunk:
    text: str
    metadata: ChunkMetadata

class IChunker(ABC):
    """Interface for document chunking implementations."""
    
    @abstractmethod
    def chunk_document(self, file_path: str) -> List[TextChunk]:
        """Chunk a document into smaller pieces with metadata."""
        pass

class IEmbeddingManager(ABC):
    """Interface for embedding management implementations."""
    
    @abstractmethod
    def add_document(self, document: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a document to the vector store."""
        pass
        
    @abstractmethod
    def query_similar(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Query for similar documents."""
        pass
        
    @abstractmethod
    def flush(self) -> None:
        """Clear all documents from the store."""
        pass

class ILLM(ABC):
    """Interface for Large Language Model implementations."""
    
    @abstractmethod
    def generate_response(self, 
                         context: str,
                         query: str,
                         system_prompt: Optional[str] = None) -> str:
        """Generate a response based on context and query."""
        pass

class BaseRAG:
    """Base class for RAG implementations."""
    
    def __init__(self,
                 llm: ILLM,
                 embedding_manager: IEmbeddingManager,
                 context_limit: int = 5):
        self.llm = llm
        self.embedding_manager = embedding_manager
        self.context_limit = context_limit
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into a context string."""
        context_parts = []
        
        for doc in documents:
            metadata_str = ", ".join(f"{k}: {v}" for k, v in doc['metadata'].items())
            context_parts.append(
                f"[Document (Distance: {doc['distance']:.4f}, {metadata_str})]\n"
                f"{doc['document']}\n"
            )
            
        return "\n".join(context_parts)
    
    def query(self, user_query: str, system_prompt: Optional[str] = None) -> str:
        """Process a user query using RAG."""
        similar_docs = self.embedding_manager.query_similar(
            query_text=user_query,
            n_results=self.context_limit
        )
        
        context = self._format_context(similar_docs)
        
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant. Use the provided context to answer the user's question.
            If the context doesn't contain relevant information, say so. Always base your answers on the provided context.
            If the context has the source name and maybe page number, mention it at the end of your response.
            Your response must be in streamlit markdown format."""
            
        return self.llm.generate_response(context, user_query, system_prompt)
