from interfaces import BaseRAG, ILLM, IEmbeddingManager
from typing import Optional
from chroma_manager import ChromaManager

class RAG(BaseRAG):
    """Unified RAG implementation that works with any LLM provider."""
    
    def __init__(self,
                 llm: ILLM,
                 embedding_manager: Optional[IEmbeddingManager] = None,
                 context_limit: int = 5,
                 use_query_preprocessing: bool = False):
        if embedding_manager is None:
            embedding_manager = ChromaManager()
            
        super().__init__(llm, embedding_manager, context_limit, use_query_preprocessing)
