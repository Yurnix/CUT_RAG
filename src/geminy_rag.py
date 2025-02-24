from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import google.generativeai as genai
from chroma_manager import ChromaManager

class GeminyRAG:
    def __init__(self, 
                chroma_manager: Optional[ChromaManager] = None,
                model: str = "gemini-2.0-flash",
                max_tokens: int = 1024,
                temperature: float = 0.7,
                context_limit: int = 5):
        """
        Initialize GeminyRAG with ChromaManager for context retrieval.
        
        Args:
            chroma_manager (Optional[ChromaManager]): ChromaManager instance. If None, creates a new one
            model (str): Gemini model to use
            max_tokens (int): Maximum tokens in response
            temperature (float): Temperature for response generation
            context_limit (int): Maximum number of similar documents to include in context
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize Gemini client
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)
        
        # Store parameters
        self.model = model
            
        # Store parameters
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
            If the context doesn't contain relevant information, say so. Always base your answers on the provided context.
            Be concise and informative. Try to provide a helpful response to the user's question.
            If the context has the source name and maybe page number, mention it at the end of your response.
            Your response must be in streamlit markdown format."""
        
        # Create message for the conversation
        message = f"""Context:
        {context}
        
        Question: {user_query}"""
        print("### GEMINI #########################################################\n")

        # Generate response using Gemini
        model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
        )
        
        # Combine system prompt and message
        combined_prompt = f"{system_prompt}\n\n{message}"
        print(combined_prompt)

        response = model.generate_content(combined_prompt)
        result = response.text
        
        print("#########################################################\n")
        print(result)
        return result
