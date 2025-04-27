from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
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
                 context_limit: int = 5,
                 use_query_preprocessing: bool = False):
        self.llm = llm
        self.embedding_manager = embedding_manager
        self.context_limit = context_limit
        self.use_query_preprocessing = use_query_preprocessing
        self.query_preprocessor = None
        self.selected_topics = []  # Default to no specific topics (use default collection)
        self.results_per_topic = 2  # Default number of results to fetch per topic
        
        # If query preprocessing is enabled, initialize the preprocessor
        if self.use_query_preprocessing:
            from query_preprocessor import QueryPreprocessor
            self.query_preprocessor = QueryPreprocessor(llm)
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into a context string."""
        context_parts = []
        
        for doc in documents:
            metadata_str = ", ".join(f"{k}: {v}" for k, v in doc['metadata'].items())
            # Include collection/topic name if available
            collection_info = f", Topic: {doc.get('collection', 'default')}" if 'collection' in doc else ""
            
            context_parts.append(
                f"[Document (Distance: {doc['distance']:.4f}{collection_info}, {metadata_str})]\n"
                f"{doc['document']}\n"
            )
            
        return "\n".join(context_parts)
    
    def set_selected_topics(self, topics: List[str], results_per_topic: int = None):
        """
        Set the topics to query from.
        
        Args:
            topics (List[str]): List of topic names that correspond to collection names
            results_per_topic (int, optional): Number of results to retrieve per topic
        """
        self.selected_topics = topics
        if results_per_topic is not None:
            self.results_per_topic = results_per_topic
    
    def query(self, user_query: str, system_prompt: Optional[str] = None, history: str = "", 
              chat_history: List[Tuple[str, str]] = None) -> str:
        """Process a user query using RAG."""
        search_query = user_query
        
        # Use query preprocessing if enabled, even for the first query with no history
        if self.use_query_preprocessing and self.query_preprocessor:
            print("\n----- QUERY PREPROCESSING -----")
            print(f"Input query: {user_query}")
            print(f"Chat history: {str(chat_history)[:100] + '...' if chat_history and len(str(chat_history)) > 100 else str(chat_history)}")
            
            search_query = self.query_preprocessor.enrich_query(
                query=user_query,
                chat_history=chat_history
            )
            
            print(f"Output enriched query: {search_query}")
            print("--------------------------------\n")
        
        # Get similar documents using the (potentially enriched) query
        if self.selected_topics:
            print(f"Querying selected topics: {', '.join(self.selected_topics)}")
            similar_docs = self.embedding_manager.query_similar(
                query_text=search_query,
                n_results=self.context_limit,
                collection_names=self.selected_topics,
                results_per_collection=self.results_per_topic
            )
        else:
            # If no topics selected, use default collection
            similar_docs = self.embedding_manager.query_similar(
                query_text=search_query,
                n_results=self.context_limit
            )
        
        context = self._format_context(similar_docs)
        
        if system_prompt is None:
            system_prompt = """You are a helpful multilingual AI assistant. Use the provided context to answer the user's question.
            If the context doesn't contain relevant information, say so. Always base your answers on the provided context.
            If the context has the source name and maybe page number, mention it at the end of your response.
            
            Important: 
            - Detect the language of the user's original question
            - Respond in the SAME LANGUAGE as the user's original question
            - Your response must be in streamlit markdown format"""
        
        # If history is provided, include it in the context
        if history:
            context = f"{history}\n\n{context}"
        
        # Create the query that includes instructions to respond in the original language
        query_with_language_instruction = f"""Question: {user_query}

Important: Respond in the same language as my question."""
            
        # Generate the response
        print("\n----- LLM RESPONSE GENERATION -----")
        print(f"Context length: {len(context)} characters")
        print(f"User query: {user_query}")
        
        response = self.llm.generate_response(context, query_with_language_instruction, system_prompt)
        
        print(f"Response: {response[:100] + '...' if len(response) > 100 else response}")
        print("-----------------------------------\n")
        
        return response
