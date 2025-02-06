from chroma_manager import ChromaManager

def main():
    # Initialize ChromaManager
    manager = ChromaManager()
    
    # Add some sample documents
    doc1 = "The quick brown fox jumps over the lazy dog"
    doc2 = "A lazy dog sleeps in the sun"
    doc3 = "The brown fox is quick and clever"
    
    # Add documents with metadata
    manager.add_document(doc1, metadata={"type": "sentence", "animal": "fox"})
    manager.add_document(doc2, metadata={"type": "sentence", "animal": "dog"})
    manager.add_document(doc3, metadata={"type": "sentence", "animal": "fox"})
    
    # Print collection stats
    print("Collection stats:", manager.get_collection_stats())
    
    # Query similar documents
    query = "quick fox"
    results = manager.query_similar(query, n_results=2)
    
    print(f"\nQuery: '{query}'")
    print("Similar documents:")
    for result in results:
        print(f"\nDocument: {result['document']}")
        print(f"Metadata: {result['metadata']}")
        print(f"Distance: {result['distance']}")
    
    # Flush the collection
    manager.flush()
    print("\nAfter flushing:")
    print("Collection stats:", manager.get_collection_stats())

if __name__ == "__main__":
    main()
