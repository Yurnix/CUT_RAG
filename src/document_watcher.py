import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
from embedding_manager import EmbeddingManager
from pdf_chunker import PdfChunker
from interfaces import IChunker
from typing import Optional
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DocumentHandler(FileSystemEventHandler):
    def __init__(self, embedding_manager: EmbeddingManager, chunker: Optional[IChunker] = None, flush_database: bool = False):
        """
        Initialize document handler with EmbeddingManager.
        
        Args:
            embedding_manager (EmbeddingManager): Instance of EmbeddingManager for document processing
        """
        self.embedding_manager = embedding_manager
        self.processed_files = set()  # Track processed files to avoid duplicates
        self.chunker = chunker or PdfChunker()  # Use PdfChunker by default
        if flush_database:
            self.embedding_manager.flush_db()
        
    def _is_supported_file(self, file_path: str) -> bool:
        """Check if the file type is supported."""
        supported_extensions = {'.txt', '.pdf', '.csv'}
        return os.path.splitext(file_path)[1].lower() in supported_extensions
        
    def _process_file(self, file_path: str):
        """Process a file using EmbeddingManager or PdfChunker for PDFs if enabled."""
        try:
            if not self._is_supported_file(file_path):
                logging.info(f"Skipping unsupported file: {file_path}")
                return
                
            if file_path in self.processed_files:
                logging.info(f"File already processed: {file_path}")
                return
                
            logging.info(f"Processing file: {file_path}")
            
            # Process the file using the appropriate chunker
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
    def __init__(self, watch_directory: str = "Docs", chunker: Optional[IChunker] = None, 
                 flush_database: bool = False, embed_existing: bool = False):
        """
        Initialize document watcher service.
        
        Args:
            watch_directory (str): Directory to monitor for changes
            chunker (Optional[IChunker]): Optional custom chunker implementation
            flush_database (bool): Whether to flush the database on start
            embed_existing (bool): Whether to embed existing files in the directory
        """
        self.watch_directory = watch_directory
        self.embed_existing = embed_existing
        
        # Initialize managers and ensure collection exists
        self.embedding_manager = EmbeddingManager()
        # Ensure collection exists by accessing it
        _ = self.embedding_manager.chroma_manager.create_collection()
        
        self.event_handler = DocumentHandler(self.embedding_manager, chunker, flush_database)
        self.observer = Observer()
        
    def _list_existing_files(self):
        """List existing files in the watch directory."""
        if not os.path.exists(self.watch_directory):
            os.makedirs(self.watch_directory)
            logging.info(f"Created directory: {self.watch_directory}")
            return
            
        supported_files = []
        unsupported_files = []
        
        for filename in os.listdir(self.watch_directory):
            file_path = os.path.join(self.watch_directory, filename)
            if os.path.isfile(file_path):
                if self.event_handler._is_supported_file(file_path):
                    supported_files.append(filename)
                else:
                    unsupported_files.append(filename)
        
        if supported_files:
            logging.info("Found supported files:")
            for filename in supported_files:
                logging.info(f"  - {filename}")
        
        if unsupported_files:
            logging.info("Found unsupported files:")
            for filename in unsupported_files:
                logging.info(f"  - {filename}")
                
        return supported_files, unsupported_files
    
    def _process_existing_files(self):
        """Process existing files if embed_existing is True."""
        if not os.path.exists(self.watch_directory):
            os.makedirs(self.watch_directory)
            logging.info(f"Created directory: {self.watch_directory}")
            return
            
        supported_files, _ = self._list_existing_files()
        
        if self.embed_existing and supported_files:
            logging.info("Embedding existing files...")
            for filename in supported_files:
                file_path = os.path.join(self.watch_directory, filename)
                self.event_handler._process_file(file_path)
        
    def start(self):
        """Start the document watcher service."""
        # Process or list existing files
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Document Watcher Service')
    parser.add_argument('--flush_database', action='store_true', help='Flush the ChromaDB before starting')
    parser.add_argument('--embed_existing', action='store_true', help='Embed existing files in the directory')
    args = parser.parse_args()
    
    watcher = DocumentWatcher(flush_database=args.flush_database, embed_existing=args.embed_existing)
    watcher.start()
