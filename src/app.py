import streamlit as st
import os
import argparse
from collections import deque
import glob
from embedding_manager import EmbeddingManager
from rag_implementations import RAG
from llm_implementations import AnthropicLLM, GeminiLLM, DeepseekLLM

# Parse command-line arguments
parser = argparse.ArgumentParser(description="RAG Chat Interface")
parser.add_argument("--history_depth", type=int, default=2, help="Number of past interactions to include in chat history")
args = parser.parse_args()

# Initialize session state for chat history and settings
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = deque(maxlen=args.history_depth)
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "anthropic"
if "selected_topics" not in st.session_state:
    st.session_state.selected_topics = []
if "results_per_topic" not in st.session_state:
    st.session_state.results_per_topic = 2

# Initialize managers
@st.cache_resource
def init_managers():
    embedding_manager = EmbeddingManager()
    anthropic_llm = AnthropicLLM()
    gemini_llm = GeminiLLM()
    deepseek_llm = DeepseekLLM()
    
    # Create RAG instances with query preprocessing enabled
    anthropic_rag = RAG(
        llm=anthropic_llm, 
        embedding_manager=embedding_manager.chroma_manager,
        use_query_preprocessing=True
    )
    gemini_rag = RAG(
        llm=gemini_llm, 
        embedding_manager=embedding_manager.chroma_manager,
        use_query_preprocessing=True
    )
    deepseek_rag = RAG(
        llm=deepseek_llm, 
        embedding_manager=embedding_manager.chroma_manager,
        use_query_preprocessing=True
    )
    
    return embedding_manager, anthropic_rag, gemini_rag, deepseek_rag

def get_topic_directories():
    """Get a list of topic directories inside Docs folder"""
    topic_dirs = []
    if os.path.exists("Docs"):
        for item in os.listdir("Docs"):
            if os.path.isdir(os.path.join("Docs", item)):
                topic_dirs.append(item)
    return topic_dirs

def main():
    st.set_page_config(page_title="Cyprus University of Technology RAG", layout="wide")
    
    # Title and description
    st.title("Cyprus University of Technology RAG")
    st.markdown("""
    This is a Multilingual Retrieval-Augmented Generation (RAG) system that allows you to 
    ask questions on a specific domain documents and get AI-powered responses 
    based on the document content.
    """)
    
    # Initialize managers
    embedding_manager, anthropic_rag, gemini_rag, deepseek_rag = init_managers()
    
    # Sidebar for settings and file upload
    with st.sidebar:
        st.header("Settings")
        st.session_state.llm_provider = st.selectbox(
            "Select LLM Provider",
            ["anthropic", "gemini", "deepseek"],
            index=0 if st.session_state.llm_provider == "anthropic" else (
                1 if st.session_state.llm_provider == "gemini" else 2)
        )
        
        # Topic selection section
        st.header("Topic Selection")
        
        # Get available topics
        topics = get_topic_directories()
        
        if topics:
            st.write("Select topics to retrieve context from:")
            
            # Create checkboxes for each topic
            selected_topics = []
            for topic in topics:
                if st.checkbox(topic, value=topic in st.session_state.selected_topics):
                    selected_topics.append(topic)
            
            st.session_state.selected_topics = selected_topics
            
            # Results per topic slider
            st.session_state.results_per_topic = st.slider(
                "Results per topic",
                min_value=1,
                max_value=5,
                value=st.session_state.results_per_topic,
                help="Number of embeddings to retrieve from each selected topic"
            )
            
            if selected_topics:
                st.success(f"Selected topics: {', '.join(selected_topics)}")
            else:
                st.info("No topics selected. Will use the default collection.")
        else:
            st.info("No topic directories found. Create subdirectories in the Docs folder to enable topic selection.")
        
        st.header("Document Upload (DEMO ONLY)")
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'csv'])
        
        # Topic selection for uploaded file
        topic_for_upload = None
        if uploaded_file:
            if topics:
                topic_for_upload = st.selectbox(
                    "Select topic for this document",
                    ["Default"] + topics,
                    index=0,
                    help="The topic/collection where this document will be stored"
                )

            # Save uploaded file temporarily
            file_path = f"temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Process the file
            try:
                collection_name = None if topic_for_upload == "Default" else topic_for_upload
                doc_ids = embedding_manager.add_file(
                    file_path, 
                    metadata={'source': uploaded_file.name},
                    collection_name=collection_name
                )
                st.success(f"Successfully processed {uploaded_file.name}")
                st.info(f"Added {len(doc_ids)} chunks to {'the default collection' if collection_name is None else f'the {collection_name} collection'}")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(file_path):
                    os.remove(file_path)
    
    # Main chat interface
    st.header("Chat Interface")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Use selected RAG implementation
                if st.session_state.llm_provider == "anthropic":
                    rag = anthropic_rag
                elif st.session_state.llm_provider == "gemini":
                    rag = gemini_rag
                else:
                    rag = deepseek_rag
                    
                # Configure RAG with selected topics
                if st.session_state.selected_topics:
                    rag.set_selected_topics(
                        st.session_state.selected_topics,
                        st.session_state.results_per_topic
                    )
                else:
                    # Reset to default if no topics selected
                    rag.set_selected_topics([])
                    
                try:
                    # Format history if available
                    history_text = ""
                    if len(st.session_state.chat_history) > 0:
                        history_text = "Chat History:\n"
                        for i, (past_question, past_response) in enumerate(st.session_state.chat_history):
                            history_text += f"User: {past_question}\nAssistant: {past_response}\n\n"
                    
                    # Query with history included and pass chat_history for query preprocessing
                    response = rag.query(
                        prompt, 
                        system_prompt=None, 
                        history=history_text,
                        chat_history=list(st.session_state.chat_history) if st.session_state.chat_history else None
                    )
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Add to concise history queue (only question/answer pairs)
                    st.session_state.chat_history.append((prompt, response))
                except Exception as e:
                    error_message = f"An error has occurred: {str(e)}. Please try again later. If the error persists, contact the administrator."
                    st.error(error_message)
                    # Add error message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history.clear()
        st.rerun()

if __name__ == "__main__":
    main()
