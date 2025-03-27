from typing import List, Tuple, Optional
from interfaces import ILLM

class QueryPreprocessor:
    """
    Preprocesses user queries by using an LLM to:
    1. Translate queries from any language to English
    2. Enrich queries with context from chat history
    This improves vector database retrieval with multilingual support.
    """
    
    def __init__(self, llm: ILLM):
        """Initialize the QueryPreprocessor with an LLM implementation."""
        self.llm = llm
    
    def enrich_query(self, 
                    query: str, 
                    chat_history: List[Tuple[str, str]] = None,
                    system_prompt: Optional[str] = None) -> str:
        """
        Translate and enrich the user query for better vector search.
        
        Args:
            query: The current user query to enrich (in any language)
            chat_history: List of (user_query, assistant_response) tuples from previous interactions
            system_prompt: Optional custom system prompt for the LLM
            
        Returns:
            An English, enriched query that better captures the user's intent
        """
        if not chat_history or len(chat_history) == 0:
            # If no history is provided, just translate the query to English
            return self._translate_to_english(query)
        
        # Format the chat history
        history_text = ""
        for i, (past_question, past_response) in enumerate(chat_history):
            history_text += f"User: {past_question}\nAssistant: {past_response}\n\n"
        
        # Create the prompt for the LLM
        if not system_prompt:
            system_prompt = """You are a multilingual query assistant. Your task is to:
            
            1. Translate the user's query to English if it's in another language
            2. Analyze the conversation history and the translated query
            3. Produce an enriched search query IN ENGLISH that will help retrieve 
               relevant information from a vector database
            
            The enriched query should:
            1. Capture the core intent of the user's latest query
            2. Include relevant context from previous conversation
            3. Add relevant keywords that might help with document retrieval
            4. Maintain clarity and focus on the main question
            5. Resolve any references to previous messages (like "it", "that", etc.)
            
            Return ONLY the enriched English query text, with no additional explanations."""
        
        prompt = f"""Conversation History:
{history_text}

Latest User Query: {query}

Translate the query to English if needed, and rewrite it to create a more comprehensive English search query that will retrieve relevant information."""
        
        # Use the LLM to generate an enriched query
        enriched_query = self.llm.generate_response(context="", query=prompt, system_prompt=system_prompt)
        
        return enriched_query.strip()
    
    def _translate_to_english(self, query: str) -> str:
        """
        Translate a query to English when no conversation history is available.
        
        Args:
            query: The user query in any language
            
        Returns:
            The translated query in English
        """
        system_prompt = """You are a multilingual translator. Your task is to:
        
        1. Translate the user's query to English if it's not already in English
        2. Keep the query meaning intact
        3. Make minimal changes if the query is already in English
        
        Return ONLY the translated English query, with no additional explanations."""
        
        prompt = f"Translate this query to English if needed: \"{query}\""
        
        result = self.llm.generate_response(context="", query=prompt, system_prompt=system_prompt)
        
        return result.strip()
