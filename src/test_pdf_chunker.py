from pdf_chunker import PdfChunker
import os

def test_pdf_chunking():
    # Initialize the chunker
    chunker = PdfChunker()
    
    # Use a PDF from RawDocs
    pdf_path = "RawDocs/Computer Architecture-6th.pdf"
    
    # Create output directory if it doesn't exist
    os.makedirs("chunker_outputs", exist_ok=True)
    
    # Process the PDF
    print(f"Processing {pdf_path}...")
    chunks = chunker.chunk_pdf(pdf_path)
    
    # Write results to file
    output_path = "chunker_outputs/chunks_output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"Chunk #{i}\n")
            f.write(f"{'='*80}\n\n")
            
            # Write metadata
            f.write("METADATA:\n")
            f.write(f"File: {chunk.metadata.file_name}\n")
            f.write(f"Page: {chunk.metadata.page_number}\n")
            f.write(f"Hash: {chunk.metadata.text_hash}\n")
            
            # Write the chunk text
            f.write("\nCONTENT:\n")
            f.write(chunk.text)
            f.write("\n")
    
    print(f"Created {len(chunks)} chunks")
    print(f"Output written to {output_path}")

if __name__ == "__main__":
    test_pdf_chunking()
