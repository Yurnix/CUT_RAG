import streamlit as st
import os
import argparse
from collections import deque
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

def main():
    st.set_page_config(page_title="Cyprus University of Technology RAG", layout="wide")
    
    # Title and description
    st.title("Cyprus University of Technology RAG")
    st.markdown("""
    This is a Multilingual Retrieval-Augmented Generation (RAG) system that allows you to 
    ask questions on a spesific domain documents and get AI-powered responses 
    based on the document contentю
    """)
    
    # Initialize managers
    embedding_manager, anthropic_rag, gemini_rag, deepseek_rag = init_managers()
    
    # Sidebar for settings and file upload
    with st.sidebar:
        st.header("Settings")
        st.session_state.llm_provider = st.selectbox(
            "Select LLM Provider",
            ["anthropic", "gemini", "deepseek"],
            index=0 if st.session_state.llm_provider == "anthropic" else 1
        )
        
        st.header("Document Upload")
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'csv'])
        
        if uploaded_file:
            # Save uploaded file temporarily
            file_path = f"temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Process the file
            try:
                doc_ids = embedding_manager.add_file(file_path, metadata={'source': uploaded_file.name})
                st.success(f"Successfully processed {uploaded_file.name}")
                st.info(f"Added {len(doc_ids)} chunks to the database")
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
                    error_message = "An error has occurred, please try again later. If the error persists, contact the administrator."
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
