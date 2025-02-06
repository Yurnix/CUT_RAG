import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
from embedding_manager import EmbeddingManager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DocumentHandler(FileSystemEventHandler):
    def __init__(self, embedding_manager: EmbeddingManager):
        """
        Initialize document handler with EmbeddingManager.
        
        Args:
            embedding_manager (EmbeddingManager): Instance of EmbeddingManager for document processing
        """
        self.embedding_manager = embedding_manager
        self.processed_files = set()  # Track processed files to avoid duplicates
        
    def _is_supported_file(self, file_path: str) -> bool:
        """Check if the file type is supported."""
        supported_extensions = {'.txt', '.pdf', '.csv'}
        return os.path.splitext(file_path)[1].lower() in supported_extensions
        
    def _process_file(self, file_path: str):
        """Process a file using EmbeddingManager."""
        try:
            if not self._is_supported_file(file_path):
                logging.info(f"Skipping unsupported file: {file_path}")
                return
                
            if file_path in self.processed_files:
                logging.info(f"File already processed: {file_path}")
                return
                
            logging.info(f"Processing file: {file_path}")
            doc_ids = self.embedding_manager.add_file(
                file_path,
                metadata={'source': os.path.basename(file_path)}
            )
            self.processed_files.add(file_path)
            logging.info(f"Successfully processed {file_path}. Added {len(doc_ids)} chunks.")
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            self._process_file(event.src_path)

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            self._process_file(event.src_path)

class DocumentWatcher:
    def __init__(self, watch_directory: str = "Docs"):
        """
        Initialize document watcher service.
        
        Args:
            watch_directory (str): Directory to monitor for changes
        """
        self.watch_directory = watch_directory
        
        # Initialize managers and ensure collection exists
        self.embedding_manager = EmbeddingManager()
        # Ensure collection exists by accessing it
        _ = self.embedding_manager.chroma_manager.create_collection()
        
        self.event_handler = DocumentHandler(self.embedding_manager)
        self.observer = Observer()
        
    def start(self):
        """Start the document watcher service."""
        # Process existing files
        self._process_existing_files()
        
        # Start watching for new changes
        self.observer.schedule(self.event_handler, self.watch_directory, recursive=False)
        self.observer.start()
        logging.info(f"Started watching directory: {self.watch_directory}")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
            logging.info("Stopped watching directory")
        
        self.observer.join()
    
    def _process_existing_files(self):
        """Process any existing files in the watch directory."""
        if not os.path.exists(self.watch_directory):
            os.makedirs(self.watch_directory)
            logging.info(f"Created directory: {self.watch_directory}")
            return
            
        for filename in os.listdir(self.watch_directory):
            file_path = os.path.join(self.watch_directory, filename)
            if os.path.isfile(file_path):
                self.event_handler._process_file(file_path)

if __name__ == "__main__":
    watcher = DocumentWatcher()
    watcher.start()
