#!/usr/bin/env python3

import argparse
import os
from typing import List, Dict
import json
from chunker import Chunker, ChunkingMethod
from embedding_manager import EmbeddingManager
import statistics

"""
Accepts the following arguments:
    * input_file: Path to the input file (supports txt, pdf, csv)
    * output_dir: Directory where chunked results will be saved
    * --chunk-size: Maximum chunk size in tokens (default: 512)
    * --stride: Stride for sliding window chunking (default: 256)
    * --num-topics: Number of topics for topic-based chunking (default: 5)

For each chunking method, it generates:
    * A separate text file containing the chunks
    * Statistics for each chunk (length, word count)
    * A summary JSON file comparing all methods

Usage example:
# Basic usage
    ./src/test_chunker.py input.txt output_chunks/

# With custom parameters
    ./src/test_chunker.py input.pdf output_chunks/ --chunk-size 1024 --stride 512 --num-topics 3

The script will create:
    * {filename}_fixed_length_chunks.txt
    * {filename}_sentence_chunks.txt
    * {filename}_sliding_window_chunks.txt
    * {filename}_topic_chunks.txt
    * {filename}_entity_chunks.txt
    * {filename}_summary.json

Each output file includes:
    * Chunk statistics (number of chunks, average length, etc.)
    * Individual chunks with clear separators
    * A summary report comparing all methods

"""

def create_output_directory(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)

def get_chunk_statistics(chunks: List[str]) -> Dict:
    """Calculate statistics for the chunks."""
    chunk_lengths = [len(chunk.split()) for chunk in chunks]
    return {
        "num_chunks": len(chunks),
        "avg_chunk_length": round(statistics.mean(chunk_lengths), 2),
        "min_chunk_length": min(chunk_lengths),
        "max_chunk_length": max(chunk_lengths),
        "std_dev_length": round(statistics.stdev(chunk_lengths) if len(chunks) > 1 else 0, 2)
    }

def write_chunks_to_file(chunks: List[str], output_path: str, stats: Dict) -> None:
    """Write chunks and their statistics to a file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write statistics
        f.write("=== Chunk Statistics ===\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
        f.write("\n=== Chunks ===\n\n")
        
        # Write chunks with separators
        for i, chunk in enumerate(chunks, 1):
            f.write(f"--- Chunk {i} ---\n")
            f.write(chunk)
            f.write("\n\n")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test different text chunking methods')
    parser.add_argument('input_file', help='Path to input file (txt, pdf, or csv)')
    parser.add_argument('output_dir', help='Directory to save chunked outputs')
    parser.add_argument('--chunk-size', type=int, default=512,
                      help='Maximum chunk size in tokens (default: 512)')
    parser.add_argument('--stride', type=int, default=256,
                      help='Stride for sliding window chunking (default: 256)')
    parser.add_argument('--num-topics', type=int, default=5,
                      help='Number of topics for topic-based chunking (default: 5)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    # Create output directory
    create_output_directory(args.output_dir)
    
    # Initialize EmbeddingManager for file parsing
    em = EmbeddingManager()
    
    # Parse input file
    try:
        text = em.parse_file(args.input_file)
    except Exception as e:
        raise ValueError(f"Error parsing file: {str(e)}")
    
    # Initialize chunker
    chunker = Chunker()
    
    # Process text with each chunking method
    results = {}
    base_filename = os.path.splitext(os.path.basename(args.input_file))[0]
    
    for method in ChunkingMethod:
        # Get chunks using current method
        chunks = chunker.chunk_text(
            text,
            method=method,
            chunk_size=args.chunk_size,
            stride=args.stride,
            num_topics=args.num_topics
        )
        
        # Calculate statistics
        stats = get_chunk_statistics(chunks)
        
        # Save results
        output_file = os.path.join(args.output_dir, f"{base_filename}_{method.value}_chunks.txt")
        write_chunks_to_file(chunks, output_file, stats)
        
        # Store stats for summary
        results[method.value] = stats
    
    # Write summary report
    summary_file = os.path.join(args.output_dir, f"{base_filename}_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nChunking complete! Results saved in: {args.output_dir}")
    print(f"Summary report: {summary_file}")
    print("\nChunking Statistics Summary:")
    print("===========================")
    for method, stats in results.items():
        print(f"\n{method.upper()}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
