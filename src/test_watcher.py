import os
import time
import threading
from document_watcher import DocumentWatcher

def create_test_file(filename: str, content: str):
    """Create a test file in the Docs directory."""
    filepath = os.path.join("Docs", filename)
    with open(filepath, "w") as f:
        f.write(content)
    return filepath

def main():
    # Start the watcher in a separate thread
    watcher = DocumentWatcher()
    watcher_thread = threading.Thread(target=watcher.start)
    watcher_thread.daemon = True
    watcher_thread.start()
    
    print("Document watcher started. Creating test files...")
    time.sleep(2)  # Give the watcher time to initialize
    
    # Create a test text file
    create_test_file("test1.txt", """
    This is a test document.
    It will be automatically processed by the document watcher.
    The watcher will chunk this text and store it in ChromaDB.
    """)
    
    time.sleep(2)  # Give time for processing
    
    # Create a test CSV file
    create_test_file("test2.csv", """
    name,age,city
    John Doe,30,New York
    Jane Smith,25,Los Angeles
    Bob Johnson,35,Chicago
    """)
    
    time.sleep(2)  # Give time for processing
    
    # Modify an existing file
    create_test_file("test1.txt", """
    This is a modified test document.
    The watcher should detect this change and reprocess the file.
    """)
    
    print("\nTest files have been created and modified.")
    print("Press Ctrl+C to stop the watcher...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCleaning up test files...")
        # Clean up test files
        for filename in ["test1.txt", "test2.csv"]:
            filepath = os.path.join("Docs", filename)
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == "__main__":
    main()
