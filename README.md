# Cyprus University of Technology RAG System

A Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask questions about them using AI.

## Features

- Document upload support (TXT, PDF, CSV)
- Intelligent text chunking and embedding (Coming soon)
- AI-powered question answering using Anthropic's Claude
- Interactive chat interface
- Document context retrieval

## Setup

1. Install main dependencies: 
   - Python 3.12 
   - [Poetry](https://python-poetry.org/docs/#installation)


2. Install project dependencies in project:
```bash
export POETRY_VIRTUALENVS_IN_PROJECT=true
poetry install --no-root
```

3. Create a `.env` file in the root directory and add your Anthropic API key (this file is gitignored):
```
ANTHROPIC_API_KEY=your-api-key-here
```

## Running the Application

1. Activate the virtual environment:
```bash
source .venv/bin/activate  # On Windows: venv\Scripts\activate
```
2. Run the watchdog for the document embedding:
```bash
python src/document_watcher.py
```
3. Run the Streamlit app:
```bash
streamlit run src/app.py
```

## Usage

1. Upload documents using the sidebar
2. Ask questions in the chat interface
3. Get AI-powered responses based on the document content

## Components

- `ChromaManager`: Handles document storage and retrieval
- `EmbeddingManager`: Processes and chunks documents
- `AnthropicRAG`: Implements the RAG interface using Claude
- Streamlit interface for user interaction

## Project Structure

```
.
├── src/                     # Source code directory
│   ├── app.py               # Streamlit web interface
│   ├── anthropic_rag.py     # RAG implementation with Claude
│   ├── chroma_manager.py    # Vector database management
│   ├── document_watcher.py  # File monitoring service
│   └── embedding_manager.py # Document processing
├── Docs/                    # Directory for documents to be processed
├── requirements.txt         # Project dependencies
├── .env                     # Environment variables (gitignored)
└── .gitignore               # Git ignore rules
```

## Development

- The project uses ChromaDB for vector storage
- Documents are automatically processed when added to the Docs/ directory
- The system supports TXT, PDF, and CSV files
- All database files are stored in chroma_db/ (gitignored)

## Notes

- Keep your API key secure and never commit it to version control
- The .env file and chroma_db/ directory are gitignored
- Virtual environment (venv/) is excluded from version control
- The document watcher service can be started separately: `python src/document_watcher.py`
