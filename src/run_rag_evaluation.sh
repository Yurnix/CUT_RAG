#!/bin/bash

# Run RAG Quality Evaluator Script
# This script provides an easy way to run the RAG Quality Evaluator with different configurations

# Display usage information
function show_usage {
    echo "RAG Quality Evaluator Runner"
    echo ""
    echo "Usage: ./run_rag_evaluation.sh [options]"
    echo ""
    echo "Options:"
    echo "  -s, --samples NUM      Number of random documents to sample (default: 5)"
    echo "  -q, --questions NUM    Questions per document (default: 2)"
    echo "  -o, --output FILE      Output file path (default: rag_evaluation_results.json)"
    echo "  -m, --model MODEL      Anthropic model to use (default: claude-3-5-sonnet-20241022)"
    echo "  -c, --collection NAME  ChromaDB collection name to evaluate (default: Computer_Architecture)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_rag_evaluation.sh"
    echo "  ./run_rag_evaluation.sh --samples 10 --questions 3"
    echo "  ./run_rag_evaluation.sh -s 3 -q 5 -o custom_evaluation.json"
    echo "  ./run_rag_evaluation.sh --collection RFC"
    echo ""
}

# Default values
SAMPLES=5
QUESTIONS=2
OUTPUT="rag_evaluation_results.json"
MODEL="claude-3-5-sonnet-20241022"
COLLECTION="documents"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--samples)
            SAMPLES="$2"
            shift 2
            ;;
        -q|--questions)
            QUESTIONS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -c|--collection)
            COLLECTION="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Verify Python environment
if ! command -v python3 &>/dev/null; then
    echo "Error: Python 3 is required but not found in PATH"
    exit 1
fi

# Check for .env file with API key
if [ ! -f ".env" ]; then
    echo "Warning: No .env file found. Make sure you have set ANTHROPIC_API_KEY environment variable."
fi

# Display configuration
echo "=========================================="
echo "Starting RAG Quality Evaluation with:"
echo "  Samples:     $SAMPLES"
echo "  Questions:   $QUESTIONS"
echo "  Output:      $OUTPUT"
echo "  Model:       $MODEL"
echo "  Collection:  $COLLECTION"
echo "=========================================="

# Run the evaluator
python3 src/rag_quality_evaluator.py --samples "$SAMPLES" --questions "$QUESTIONS" --output "$OUTPUT" --model "$MODEL" --collection "$COLLECTION"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "Evaluation completed successfully!"
    echo "Results saved to: $OUTPUT"
    echo "Log file: rag_evaluation.log"
    echo ""
    echo "To view a summary of results:"
    echo "  cat $OUTPUT | grep -A 15 'summary'"
    echo ""
else
    echo ""
    echo "Evaluation failed. Check rag_evaluation.log for details."
    echo ""
fi
