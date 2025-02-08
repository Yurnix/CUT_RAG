from typing import List
import spacy
from transformers import AutoTokenizer
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import numpy as np
from enum import Enum

class ChunkingMethod(Enum):
    FIXED_LENGTH = "fixed_length"
    SENTENCE = "sentence"
    SLIDING_WINDOW = "sliding_window"
    TOPIC = "topic"
    ENTITY = "entity"

class Chunker:
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize Chunker with specified tokenizer model.
        
        Args:
            model_name (str): Name of the HuggingFace tokenizer model
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nlp = spacy.load("en_core_web_sm")
        
    def fixed_length_chunk(self, text: str, chunk_size: int = 512, stride: int = 0) -> List[str]:
        """
        Split text into fixed-length chunks based on token count.
        
        Args:
            text (str): Input text
            chunk_size (int): Maximum number of tokens per chunk
            stride (int): Number of overlapping tokens between chunks
            
        Returns:
            List[str]: List of text chunks
        """
        # Tokenize the text
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        # Calculate effective chunk size considering stride
        effective_chunk_size = chunk_size - stride
        
        # Split tokens into chunks
        for i in range(0, len(tokens), effective_chunk_size):
            # Get chunk tokens
            chunk_tokens = tokens[i:i + chunk_size]
            # Decode tokens back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            
        return chunks
    
    def sentence_chunk(self, text: str, max_tokens: int = 512) -> List[str]:
        """
        Split text into chunks based on sentence boundaries.
        
        Args:
            text (str): Input text
            max_tokens (int): Maximum number of tokens per chunk
            
        Returns:
            List[str]: List of text chunks
        """
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in doc.sents:
            # Get token count for current sentence
            sent_tokens = len(self.tokenizer.encode(sent.text))
            
            # If adding this sentence exceeds max_tokens, save current chunk and start new one
            if current_length + sent_tokens > max_tokens:
                if current_chunk:
                    chunks.append(" ".join([s.text for s in current_chunk]))
                current_chunk = [sent]
                current_length = sent_tokens
            else:
                current_chunk.append(sent)
                current_length += sent_tokens
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(" ".join([s.text for s in current_chunk]))
            
        return chunks
    
    def sliding_window_chunk(self, text: str, window_size: int = 512, stride: int = 256) -> List[str]:
        """
        Split text into overlapping chunks using sliding window.
        
        Args:
            text (str): Input text
            window_size (int): Size of each window in tokens
            stride (int): Number of tokens to slide the window by
            
        Returns:
            List[str]: List of text chunks
        """
        return self.fixed_length_chunk(text, chunk_size=window_size, stride=stride)
    
    def topic_chunk(self, text: str, num_topics: int = 5, chunk_size: int = 512) -> List[str]:
        """
        Split text into chunks based on topic coherence.
        
        Args:
            text (str): Input text
            num_topics (int): Number of topics to identify
            chunk_size (int): Approximate size of each chunk in tokens
            
        Returns:
            List[str]: List of text chunks
        """
        # Process text with spaCy
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        # Prepare documents for LDA
        texts = [[token.lemma_ for token in sent 
                 if not token.is_stop and not token.is_punct and token.is_alpha]
                for sent in sentences]
        
        # Create dictionary and corpus
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        # Train LDA model
        lda = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        
        # Get topic distribution for each sentence
        sentence_topics = [lda.get_document_topics(bow) for bow in corpus]
        
        # Group sentences by dominant topic
        topic_sentences = [[] for _ in range(num_topics)]
        for sent, topics in zip(sentences, sentence_topics):
            if topics:  # Check if topics list is not empty
                dominant_topic = max(topics, key=lambda x: x[1])[0]
                topic_sentences[dominant_topic].append(sent)
        
        # Create chunks based on topics
        chunks = []
        current_chunk = []
        current_length = 0
        
        for topic_group in topic_sentences:
            for sent in topic_group:
                sent_tokens = len(self.tokenizer.encode(sent.text))
                
                if current_length + sent_tokens > chunk_size:
                    if current_chunk:
                        chunks.append(" ".join([s.text for s in current_chunk]))
                    current_chunk = [sent]
                    current_length = sent_tokens
                else:
                    current_chunk.append(sent)
                    current_length += sent_tokens
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(" ".join([s.text for s in current_chunk]))
            
        return chunks
    
    def entity_chunk(self, text: str, max_tokens: int = 512) -> List[str]:
        """
        Split text into chunks based on named entity boundaries.
        
        Args:
            text (str): Input text
            max_tokens (int): Maximum number of tokens per chunk
            
        Returns:
            List[str]: List of text chunks
        """
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Get sentence spans that contain entities
        entity_sentences = set()
        for ent in doc.ents:
            sent = ent.sent
            if sent:
                entity_sentences.add(sent)
        
        # Process sentences
        for sent in doc.sents:
            sent_tokens = len(self.tokenizer.encode(sent.text))
            
            # If sentence contains entity, try to keep it in the same chunk
            if sent in entity_sentences:
                # If adding this sentence would exceed max_tokens, save current chunk
                if current_length + sent_tokens > max_tokens:
                    if current_chunk:
                        chunks.append(" ".join([s.text for s in current_chunk]))
                    current_chunk = [sent]
                    current_length = sent_tokens
                else:
                    current_chunk.append(sent)
                    current_length += sent_tokens
            else:
                # For sentences without entities, split more freely
                if current_length + sent_tokens > max_tokens:
                    if current_chunk:
                        chunks.append(" ".join([s.text for s in current_chunk]))
                    current_chunk = [sent]
                    current_length = sent_tokens
                else:
                    current_chunk.append(sent)
                    current_length += sent_tokens
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(" ".join([s.text for s in current_chunk]))
            
        return chunks
    
    def chunk_text(self, text: str, method: ChunkingMethod = ChunkingMethod.FIXED_LENGTH, 
                  chunk_size: int = 512, stride: int = 256, num_topics: int = 5) -> List[str]:
        """
        Split text into chunks using the specified method.
        
        Args:
            text (str): Text to chunk
            method (ChunkingMethod): Chunking method to use
            chunk_size (int): Size of chunks in tokens
            stride (int): Stride for sliding window
            num_topics (int): Number of topics for topic-based chunking
            
        Returns:
            List[str]: List of text chunks
        """
        if method == ChunkingMethod.FIXED_LENGTH:
            return self.fixed_length_chunk(text, chunk_size)
        elif method == ChunkingMethod.SENTENCE:
            return self.sentence_chunk(text, chunk_size)
        elif method == ChunkingMethod.SLIDING_WINDOW:
            return self.sliding_window_chunk(text, chunk_size, stride)
        elif method == ChunkingMethod.TOPIC:
            return self.topic_chunk(text, num_topics, chunk_size)
        elif method == ChunkingMethod.ENTITY:
            return self.entity_chunk(text, chunk_size)
        else:
            raise ValueError(f"Unknown chunking method: {method}")
