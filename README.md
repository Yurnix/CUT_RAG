# CUT RAG System

A Retrieval-Augmented Generation (RAG) system for the Cyprus University of Technology that allows users to:
1. Upload and process documents (TXT, PDF, CSV)
2. Ask questions about the uploaded documents
3. Get AI-powered responses based on the document content

## Architecture

The project follows SOLID principles and is organized into several key components:

### Core Components

1. **Document Processing**
   - `document_watcher.py`: Monitors a directory for file changes and automatically processes documents
   - `pdf_chunker.py`: Handles PDF document chunking with metadata tracking
   - `text_chunker.py`: Handles text document chunking

2. **Vector Store Management**
   - `chroma_manager.py`: Manages ChromaDB operations and document storage
   - `embedding_manager.py`: Handles document embedding and chunk management
   - `chroma_gui.py`: GUI tool for browsing and managing ChromaDB contents

3. **RAG Implementation**
   - `rag_implementations.py`: Unified RAG implementation supporting multiple LLM providers
   - `llm_implementations.py`: LLM provider implementations (Anthropic, Gemini, Deepseek)
   - `interfaces.py`: Core interfaces and base classes

4. **Web Interface**
   - `app.py`: Streamlit web interface for document upload and querying

### Key Features

- **Multi-LLM Support**: Supports multiple LLM providers through a unified interface
- **Smart Document Processing**: 
  - Automatic file type detection and appropriate chunking
  - Efficient handling of file modifications and deletions
  - Debounced processing to prevent duplicate operations
- **ChromaDB Management**:
  - Efficient vector storage and retrieval
  - GUI tool for database inspection and management
  - Metadata-based document tracking

## Usage

1. **Start the Document Watcher**:
   ```bash
   python src/document_watcher.py [--flush_database] [--embed_existing]
   ```
   - `--flush_database`: Clear the database before starting
   - `--embed_existing`: Process existing files in the watch directory

2. **Start the Web Interface**:
   ```bash
   python src/app.py
   ```

3. **Browse ChromaDB Contents**:
   ```bash
   python src/chroma_gui.py
   ```

## Directory Structure

- `src/`: Source code files
- `Docs/`: Directory watched for documents to process
- `RawDocs/`: Original document storage
- `chroma_db/`: ChromaDB persistent storage
- `chunker_outputs/`: Temporary storage for chunking results

## Environment Setup

1. Create a `.env` file with your API keys:
   ```
   ANTHROPIC_API_KEY=your_key_here
   GOOGLE_API_KEY=your_key_here
   DEEPSEEK_API_KEY=your_key_here
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Testing

The project includes several test files:
- `test_chunker.py`: Test document chunking functionality
- `test_pdf_chunker.py`: Test PDF-specific chunking
- `test_chroma.py`: Test ChromaDB operations
- `test_embedding.py`: Test embedding functionality
- `test_rag.py`: Test RAG implementation
- `test_watcher.py`: Test document watching functionality
