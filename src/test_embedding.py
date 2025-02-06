from embedding_manager import EmbeddingManager
import os

def create_test_files():
    """Create sample files for testing"""
    
    # Create a test text file
    with open('test.txt', 'w') as f:
        f.write("This is a test document. It contains multiple sentences. " * 10)
    
    # Create a test CSV file
    with open('test.csv', 'w') as f:
        f.write("name,age,city\n")
        f.write("John Doe,30,New York\n")
        f.write("Jane Smith,25,Los Angeles\n")
        f.write("Bob Johnson,35,Chicago\n")

def main():
    # Create test files
    create_test_files()
    
    # Initialize EmbeddingManager with a small threshold for demonstration
    manager = EmbeddingManager(symbol_threshold=100)
    
    # Process text file
    print("\nProcessing text file...")
    txt_ids = manager.add_file('test.txt', metadata={'category': 'test'})
    print(f"Added {len(txt_ids)} chunks from text file")
    
    # Process CSV file
    print("\nProcessing CSV file...")
    csv_ids = manager.add_file('test.csv', metadata={'category': 'test'})
    print(f"Added {len(csv_ids)} chunks from CSV file")
    
    # Query similar documents
    query = "test document"
    print(f"\nQuerying similar documents to: '{query}'")
    results = manager.chroma_manager.query_similar(query, n_results=2)
    
    print("\nResults:")
    for result in results:
        print(f"\nChunk: {result['document'][:100]}...")
        print(f"Source: {result['metadata']['source_file']}")
        print(f"Chunk Index: {result['metadata']['chunk_index']}")
        print(f"Distance: {result['distance']}")
    
    # Clean up test files
    os.remove('test.txt')
    os.remove('test.csv')

if __name__ == "__main__":
    main()
