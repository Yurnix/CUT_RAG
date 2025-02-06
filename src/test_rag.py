from anthropic_rag import AnthropicRAG
from embedding_manager import EmbeddingManager

def main():
    # Initialize managers
    embedding_manager = EmbeddingManager()
    rag = AnthropicRAG(chroma_manager=embedding_manager.chroma_manager)
    
    # Create and add a test document
    with open('test_doc.txt', 'w') as f:
        f.write("""
        The Python programming language was created by Guido van Rossum and was first released in 1991.
        Python is known for its simplicity and readability, emphasizing a clean and straightforward syntax.
        It supports multiple programming paradigms, including procedural, object-oriented, and functional programming.
        Python has a large standard library and a vast ecosystem of third-party packages.
        """)
    
    # Add document to ChromaDB
    print("Adding document to ChromaDB...")
    embedding_manager.add_file('test_doc.txt', metadata={'subject': 'Python'})
    
    # Test queries
    questions = [
        "Who created Python and when?",
        "What are the main characteristics of Python?",
        "What is Python's relationship with JavaScript?" # This should indicate no relevant information
    ]
    
    print("\nTesting RAG queries:")
    for question in questions:
        print(f"\nQ: {question}")
        answer = rag.query(question)
        print(f"A: {answer}")
    
    # Clean up
    import os
    os.remove('test_doc.txt')

if __name__ == "__main__":
    main()
