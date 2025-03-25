import tkinter as tk
from tkinter import ttk, scrolledtext
import chromadb
from chromadb.config import Settings
import json

class ChromaDBBrowser(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("ChromaDB Browser")
        self.geometry("1200x800")

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Create main layout
        self.create_widgets()
        self.refresh_collections()

    def create_widgets(self):
        # Left panel for collections
        left_frame = ttk.Frame(self)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Label(left_frame, text="Collections:").pack(anchor=tk.W)
        self.collection_listbox = tk.Listbox(left_frame, width=30, height=20)
        self.collection_listbox.pack(fill=tk.Y, expand=True)
        self.collection_listbox.bind('<<ListboxSelect>>', self.on_collection_select)

        refresh_btn = ttk.Button(left_frame, text="Refresh Collections", command=self.refresh_collections)
        refresh_btn.pack(fill=tk.X, pady=5)

        # Right panel for document details
        right_frame = ttk.Frame(self)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Documents list
        doc_frame = ttk.Frame(right_frame)
        doc_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(doc_frame, text="Documents:").pack(anchor=tk.W)
        self.doc_listbox = tk.Listbox(doc_frame, height=10)
        self.doc_listbox.pack(fill=tk.X)
        self.doc_listbox.bind('<<ListboxSelect>>', self.on_document_select)

        # Document details
        details_frame = ttk.Frame(right_frame)
        details_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        ttk.Label(details_frame, text="Document Details:").pack(anchor=tk.W)
        self.details_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD, height=20)
        self.details_text.pack(fill=tk.BOTH, expand=True)

    def refresh_collections(self):
        self.collection_listbox.delete(0, tk.END)
        collections = self.client.list_collections()
        for collection in collections:
            self.collection_listbox.insert(tk.END, collection)

    def on_collection_select(self, event):
        selection = self.collection_listbox.curselection()
        if not selection:
            return

        collection_name = self.collection_listbox.get(selection[0])
        collection = self.client.get_collection(collection_name)
        
        # Clear previous documents
        self.doc_listbox.delete(0, tk.END)
        self.details_text.delete('1.0', tk.END)

        # Get all documents in the collection
        try:
            results = collection.get()
            for i, doc_id in enumerate(results['ids']):
                display_text = f"{doc_id[:8]}... - {results['documents'][i][:50]}..."
                self.doc_listbox.insert(tk.END, display_text)
                # Store the full data for later retrieval
                self.doc_listbox.data = results
        except Exception as e:
            self.details_text.insert(tk.END, f"Error loading documents: {str(e)}")

    def on_document_select(self, event):
        selection = self.doc_listbox.curselection()
        if not selection or not hasattr(self.doc_listbox, 'data'):
            return

        idx = selection[0]
        results = self.doc_listbox.data

        # Format document details
        details = {
            "ID": results['ids'][idx],
            "Document": results['documents'][idx],
            "Metadata": results['metadatas'][idx] if results['metadatas'] else {},
            "Embeddings": results['embeddings'][idx] if results['embeddings'] else []
        }

        # Display formatted details
        self.details_text.delete('1.0', tk.END)
        self.details_text.insert(tk.END, json.dumps(details, indent=2))

def main():
    app = ChromaDBBrowser()
    app.mainloop()

if __name__ == "__main__":
    main()
