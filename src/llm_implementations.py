import os
from typing import Optional
from dotenv import load_dotenv
from anthropic import Anthropic
import google.generativeai as genai
import deepseek
from interfaces import ILLM

class AnthropicLLM(ILLM):
    def __init__(self,
                 model: str = "claude-3-5-sonnet-20241022",
                 max_tokens: int = 1024,
                 temperature: float = 0.7):
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def generate_response(self, context: str, query: str, system_prompt: Optional[str] = None) -> str:
        message = f"""Context:
        {context}
        
        Question: {query}"""
        print(message)
        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": message}],
            system=system_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        return response.content[0].text

class GeminiLLM(ILLM):
    def __init__(self,
                 model: str = "gemini-2.0-flash",
                 max_tokens: int = 1024,
                 temperature: float = 0.7):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        genai.configure(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def generate_response(self, context: str, query: str, system_prompt: Optional[str] = None) -> str:
        combined_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}"
        
        model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
        )
        
        response = model.generate_content(combined_prompt)
        return response.text

class DeepseekLLM(ILLM):
    def __init__(self,
                 model: str = "deepseek-chat-67b",
                 max_tokens: int = 1024,
                 temperature: float = 0.7):
        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
            
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def generate_response(self, context: str, query: str, system_prompt: Optional[str] = None) -> str:
        message = f"""Context:
        {context}
        
        Question: {query}"""
        
        deepseekAPI = deepseek.DeepSeekAPI(api_key=self.api_key)
        response = deepseekAPI.chat_completion(
            prompt=message,
            prompt_sys=system_prompt,
            model=self.model,
            stream=False
        )
        
        return response.choices[0].message.content
