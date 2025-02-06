from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from anthropic import Anthropic
from chroma_manager import ChromaManager

class AnthropicRAG:
    def __init__(self, 
                chroma_manager: Optional[ChromaManager] = None,
                model: str = "claude-3-opus-20240229",
                max_tokens: int = 1024,
                temperature: float = 0.7,
                context_limit: int = 5):
        """
        Initialize AnthropicRAG with ChromaManager for context retrieval.
        
        Args:
            chroma_manager (Optional[ChromaManager]): ChromaManager instance. If None, creates a new one
            model (str): Anthropic model to use
            max_tokens (int): Maximum tokens in response
            temperature (float): Temperature for response generation
            context_limit (int): Maximum number of similar documents to include in context
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            
        # Initialize Anthropic client
        self.client = Anthropic(api_key=api_key)
        
        # Store parameters
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.context_limit = context_limit
        
        # Initialize or store ChromaManager
        self.chroma_manager = chroma_manager or ChromaManager()
        
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a context string.
        
        Args:
            documents (List[Dict[str, Any]]): List of documents from ChromaManager
            
        Returns:
            str: Formatted context string
        """
        context_parts = []
        
        for doc in documents:
            # Format metadata
            metadata_str = ", ".join(f"{k}: {v}" for k, v in doc['metadata'].items())
            
            # Add document with metadata
            context_parts.append(
                f"[Document (Distance: {doc['distance']:.4f}, {metadata_str})]\n"
                f"{doc['document']}\n"
            )
            
        return "\n".join(context_parts)
        
    def query(self, user_query: str, system_prompt: Optional[str] = None) -> str:
        """
        Process a user query using RAG.
        
        Args:
            user_query (str): User's question or query
            system_prompt (Optional[str]): Optional system prompt to guide the response
            
        Returns:
            str: Generated response
        """
        # Retrieve relevant documents
        similar_docs = self.chroma_manager.query_similar(
            query_text=user_query,
            n_results=self.context_limit
        )
        
        # Format context
        context = self._format_context(similar_docs)
        
        # Prepare system prompt
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant. Use the provided context to answer the user's question.
            If the context doesn't contain relevant information, say so. Always base your answers on the provided context."""
        
        # Create message for the conversation
        message = f"""Context:
        {context}
        
        Question: {user_query}"""
        
        # Generate response
        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": message}],
            system=system_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        return response.content[0].text
