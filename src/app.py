import streamlit as st
import os
from embedding_manager import EmbeddingManager
from anthropic_rag import AnthropicRAG

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize managers
@st.cache_resource
def init_managers():
    embedding_manager = EmbeddingManager()
    rag = AnthropicRAG(chroma_manager=embedding_manager.chroma_manager)
    return embedding_manager, rag

def main():
    st.set_page_config(page_title="Cyprus University of Technology RAG", layout="wide")
    
    # Title and description
    st.title("Cyprus University of Technology RAG")
    st.markdown("""
    This is a Retrieval-Augmented Generation (RAG) system that allows you to:
    1. Upload documents (TXT, PDF, CSV)
    2. Ask questions about the uploaded documents
    3. Get AI-powered responses based on the document content
    """)
    
    # Initialize managers
    embedding_manager, rag = init_managers()
    
    # Sidebar for file upload
    with st.sidebar:
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
                response = rag.query(prompt)
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
