import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
from embedding_manager import EmbeddingManager
from pdf_chunker import PdfChunker
from text_chunker import TextChunker
from interfaces import IChunker
from typing import Optional, Dict
import logging
import argparse
from threading import Timer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DocumentHandler(FileSystemEventHandler):
    def __init__(self, embedding_manager: EmbeddingManager, chunker: Optional[IChunker] = None):
        """
        Initialize document handler with EmbeddingManager.
        
        Args:
            embedding_manager (EmbeddingManager): Instance of EmbeddingManager for document processing
        """
        self.embedding_manager = embedding_manager
        self.pdf_chunker = chunker or PdfChunker()
        self.text_chunker = TextChunker()
        self.processing_timers: Dict[str, Timer] = {}
        self.DEBOUNCE_SECONDS = 1  # Wait for 1 second of no events before processing
        
    def _is_supported_file(self, file_path: str) -> bool:
        """Check if the file type is supported."""
        supported_extensions = {'.txt', '.pdf', '.csv'}
        return os.path.splitext(file_path)[1].lower() in supported_extensions
    
    def _get_chunker_for_file(self, file_path: str) -> IChunker:
        """Get appropriate chunker based on file type."""
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.pdf':
            return self.pdf_chunker
        else:
            return self.text_chunker
    
    def _remove_file_chunks(self, file_path: str):
        """Remove all chunks associated with a file."""
        file_name = os.path.basename(file_path)
        logging.info(f"Removing existing chunks for {file_name}")
        self.embedding_manager.chroma_manager.delete_documents_by_metadata('source', file_name)
        
    def _process_file(self, file_path: str):
        """Process a file using EmbeddingManager."""
        try:
            if not self._is_supported_file(file_path):
                logging.info(f"Skipping unsupported file: {file_path}")
                return
            
            file_name = os.path.basename(file_path)
            
            # Always remove existing chunks first
            self._remove_file_chunks(file_path)
            
            logging.info(f"Processing file: {file_path}")
            
            # Get chunks using appropriate chunker
            chunker = self._get_chunker_for_file(file_path)
            chunks = chunker.chunk_document(file_path)
            
            # Process each chunk
            doc_ids = []
            for chunk in chunks:
                chunk_id = self.embedding_manager.add_file(
                    file_path,
                    metadata={
                        'source': file_name
                    },
                    text_content=chunk.text
                )
                doc_ids.extend(chunk_id)
            
            logging.info(f"Successfully processed {file_path}. Added {len(doc_ids)} chunks.")
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
    
    def _debounced_process_file(self, file_path: str):
        """Process file with debouncing to prevent duplicate processing."""
        # Cancel any existing timer for this file
        if file_path in self.processing_timers:
            self.processing_timers[file_path].cancel()
            
        # Create new timer
        timer = Timer(self.DEBOUNCE_SECONDS, self._process_file, args=[file_path])
        self.processing_timers[file_path] = timer
        timer.start()

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            self._debounced_process_file(event.src_path)

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            self._debounced_process_file(event.src_path)
    
    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory:
            # Cancel any pending processing
            if event.src_path in self.processing_timers:
                self.processing_timers[event.src_path].cancel()
                del self.processing_timers[event.src_path]
            self._remove_file_chunks(event.src_path)

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
        
        # Ensure collection exists and handle database flushing
        _ = self.embedding_manager.chroma_manager.create_collection()
        if flush_database:
            logging.info("Flushing database...")
            self.embedding_manager.flush_db()
        
        self.event_handler = DocumentHandler(self.embedding_manager, chunker)
        self.observer = Observer()
        
    def _list_existing_files(self):
        """List existing files in the watch directory."""
        if not os.path.exists(self.watch_directory):
            os.makedirs(self.watch_directory)
            logging.info(f"Created directory: {self.watch_directory}")
            return [], []
            
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
                # Process existing files directly without debouncing
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
            # Cancel any pending timers
            for timer in self.event_handler.processing_timers.values():
                timer.cancel()
            logging.info("Stopped watching directory")
        
        self.observer.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Document Watcher Service')
    parser.add_argument('--flush_database', action='store_true', help='Flush the ChromaDB before starting')
    parser.add_argument('--embed_existing', action='store_true', help='Embed existing files in the directory')
    args = parser.parse_args()
    
    watcher = DocumentWatcher(flush_database=args.flush_database, embed_existing=args.embed_existing)
    watcher.start()
