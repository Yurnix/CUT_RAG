# RAG Quality Evaluator

This tool automatically evaluates the quality of responses from your Retrieval-Augmented Generation (RAG) system by:

1. Selecting random documents from your ChromaDB
2. Generating test questions for these documents using Anthropic API
3. Processing questions through your RAG system
4. Evaluating the quality of responses using Anthropic API

## Features

- Completely automated evaluation pipeline
- Detailed logging of each step in the process
- Random document sampling from ChromaDB
- AI-generated questions tailored to document content
- Objective quality scoring (0-10 scale)
- Qualitative feedback on response quality
- Summary statistics of evaluation results
- JSON output for further analysis

## Requirements

- Python 3.7+
- ChromaDB with indexed documents
- Anthropic API key in `.env` file

## Setup

Ensure your `.env` file contains your Anthropic API key:

```
ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

Run the evaluator with default settings:

```bash
python rag_quality_evaluator.py
```

### Command Line Arguments

- `--samples N`: Number of random documents to select (default: 5)
- `--questions N`: Number of questions per document (default: 2)
- `--output PATH`: Output JSON file path (default: rag_evaluation_results.json)
- `--model MODEL`: Anthropic model to use (default: claude-3-5-sonnet-20241022)
- `--collection NAME`: ChromaDB collection name to evaluate (default: documents)

Example with custom settings:

```bash
python rag_quality_evaluator.py --samples 10 --questions 3 --output custom_evaluation.json --collection documents
```

## Output

The script generates:

1. A JSON file with full evaluation results
2. A log file (`rag_evaluation.log`) with detailed process information
3. A console summary of evaluation scores

### JSON Output Format

The output JSON contains:

- `metadata`: Information about the evaluation run
- `evaluations`: Detailed results for each question, including:
  - The original document text
  - The generated question
  - The RAG system's response
  - The evaluation score (0-10)
  - An explanation of the score
- `summary`: Statistical overview of results

## Understanding Scores

The evaluation scores range from 0 to 10:

- **0-3**: Poor quality - Incorrect, irrelevant, or misleading responses
- **4-6**: Moderate quality - Partially correct but with significant issues
- **7-8**: Good quality - Generally correct with minor omissions
- **9-10**: Excellent quality - Comprehensive, accurate, and well-explained

## Troubleshooting

- Check `rag_evaluation.log` for detailed error messages
- Ensure your ChromaDB contains documents
- Verify your Anthropic API key is correct and has sufficient quota
- If document sampling fails, try with a smaller `--samples` value
