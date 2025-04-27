"""
RAG System Quality Evaluator

This script automates the evaluation of RAG system response quality by:
1. Selecting random documents from ChromaDB
2. Generating questions for each document using Anthropic API
3. Processing these questions through the RAG system
4. Evaluating the quality of responses using Anthropic API
"""

import random
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import os
import datetime
from dotenv import load_dotenv
import time


from chroma_manager import ChromaManager
from rag_implementations import RAG
from llm_implementations import AnthropicLLM
from interfaces import ILLM, IEmbeddingManager


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("rag_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RAGQualityEvaluator:
    """
    A class to evaluate the quality of RAG system responses.
    
    This class selects random documents from ChromaDB, generates questions,
    processes them through the RAG system, and evaluates the quality of responses.
    """
    
    def __init__(
        self,
        llm: ILLM,
        embedding_manager: Optional[IEmbeddingManager] = None,
        num_samples: int = 5,
        questions_per_document: int = 2,
        results_per_topic: int = 2,
        evaluation_output_path: str = "rag_evaluation_results.json",
        collection_name: str = "Computer_Architecture"
    ):
        """
        Initialize the RAG quality evaluator.
        
        Args:
            llm: Large Language Model implementation for generating questions and evaluations
            embedding_manager: Vector database manager (defaults to ChromaManager)
            num_samples: Number of random documents to select from the database
            questions_per_document: Number of questions to generate per document
            evaluation_output_path: Path to save evaluation results
        """
        self.llm = llm
        self.embedding_manager = embedding_manager or ChromaManager()
        self.num_samples = num_samples
        self.questions_per_document = questions_per_document
        self.evaluation_output_path = evaluation_output_path
        self.collection_name = collection_name
        self.results_per_topic = results_per_topic
        self.rag = RAG(llm=self.llm, embedding_manager=self.embedding_manager)
        # Configure RAG to use the specified collection with the specified number of results
        self.rag.set_selected_topics([self.collection_name], self.results_per_topic)
        self.results = {
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "num_samples": num_samples,
                "questions_per_document": questions_per_document,
                "results_per_topic": results_per_topic,
                "collection_name": collection_name
            },
            "evaluations": []
        }
    
    def select_random_documents(self) -> List[Dict[str, Any]]:
        """
        Select random documents from the specified ChromaDB collection.
        
        Returns:
            List of document dictionaries
        """
        logger.info(f"Selecting {self.num_samples} random documents from ChromaDB collection '{self.collection_name}'...")
        
        # Get collection info to determine total document count
        stats = self.embedding_manager.get_collection_stats(collection_name=self.collection_name)
        total_documents = stats.get("total_documents", 0)
        
        if total_documents == 0:
            logger.error("No documents found in the collection.")
            return []
        
        # Get all documents from the collection
        # Note: This is a simple approach - for very large collections,
        # you might want to implement more efficient sampling
        try:
            collection = self.embedding_manager.create_collection(self.collection_name)
            all_ids = collection.get()["ids"]
            
            # Randomly select document IDs
            if len(all_ids) <= self.num_samples:
                selected_ids = all_ids
                logger.warning(f"Only {len(all_ids)} documents available, using all of them.")
            else:
                selected_ids = random.sample(all_ids, self.num_samples)
            
            # Get the selected documents
            selected_docs = []
            for doc_id in selected_ids:
                result = collection.get(ids=[doc_id])
                
                selected_docs.append({
                    'id': doc_id,
                    'document': result["documents"][0],
                    'metadata': result["metadatas"][0] if result["metadatas"] else {}
                })
            
            logger.info(f"Successfully selected {len(selected_docs)} documents")
            return selected_docs
            
        except Exception as e:
            logger.error(f"Error selecting documents: {str(e)}")
            return []
    
    def generate_questions(self, document: Dict[str, Any]) -> List[str]:
        """
        Generate questions for a given document using Anthropic API.
        
        Args:
            document: Document dictionary containing text and metadata
            
        Returns:
            List of generated questions
        """
        doc_text = document['document']
        doc_metadata = document['metadata']
        
        logger.info(f"Generating {self.questions_per_document} questions for document {document['id']}")
        
        # Create a prompt for question generation
        system_prompt = """
        You are an expert at creating insightful and diverse questions about text passages.
        Given a passage of text (which may be a chunk from a textbook, article, or document),
        your task is to generate thoughtful questions that:
        
        1. Test understanding of the key concepts in the passage
        2. Require information specifically contained in the passage to answer correctly
        3. Are clear, concise, and unambiguous
        4. Cover different aspects and difficulty levels
        5. Would be appropriate for a computer engineering student
        
        Focus on generating questions where the information in the passage is sufficient to provide a good answer.
        Important:
        - The RAG system doesn't know on what passage you are basing your questions, it will have to retrieve the correct passage to answer them.
        - Don't base your questions on figures or tables, they may not be available to the RAG system.
        """
        
        user_prompt = f"""
        Please generate {self.questions_per_document} insightful questions based on the following passage.
        Each question should be answerable using the information in this passage.
        
        PASSAGE:
        {doc_text}
        
        METADATA:
        {json.dumps(doc_metadata, indent=2)}
        
        FORMAT YOUR RESPONSE AS A JSON LIST OF STRINGS, ONLY THE QUESTIONS WITH NO ADDITIONAL TEXT.
        Example: ["Question 1?", "Question 2?"]
        """
        
        try:
            # Generate questions using the LLM
            response = self.llm.generate_response(
                context="",  # Not needed for this generation
                query=user_prompt,
                system_prompt=system_prompt
            )
            
            # Parse the response as JSON
            # The response should be a JSON list of strings
            # Try to handle various response formats
            try:
                # First try to parse the entire response as JSON
                questions = json.loads(response)
                
                # Ensure it's a list
                if not isinstance(questions, list):
                    logger.warning("Response was not a list, attempting fallback parsing")
                    # If it's not a list, try to find a JSON array in the response
                    import re
                    json_match = re.search(r'\[.*\]', response, re.DOTALL)
                    if json_match:
                        questions = json.loads(json_match.group(0))
                    else:
                        # Last resort: split by newlines and clean up
                        questions = [q.strip() for q in response.split('\n') if q.strip()]
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract questions manually
                logger.warning("JSON parsing failed, attempting manual extraction")
                import re
                # Look for questions with question marks
                questions = re.findall(r'(?:"([^"]+\?)")|(?:(?<=\n)([^"\n]+\?))', response)
                questions = [q[0] if q[0] else q[1] for q in questions if q[0] or q[1]]
                
                if not questions:
                    # Split by newlines and look for lines that end with question marks
                    lines = response.split('\n')
                    questions = [line.strip() for line in lines if line.strip().endswith('?')]
            
            # Limit to the requested number of questions
            questions = questions[:self.questions_per_document]
            
            logger.info(f"Generated {len(questions)} questions")
            logger.debug(f"Questions: {questions}")
            
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return [f"What is the main topic of this passage?", 
                    f"Can you explain the key concepts in this text?"]
    
    def process_questions_with_rag(self, document: Dict[str, Any], questions: List[str]) -> List[Dict[str, Any]]:
        """
        Process questions through the RAG system.
        
        Args:
            document: Original document dictionary
            questions: List of questions to process
            
        Returns:
            List of dictionaries containing questions and their RAG responses
        """
        results = []
        
        for question in questions:
            logger.info(f"Processing question: {question}")
            
            try:
                # Get RAG response
                rag_response = self.rag.query(question)
                
                logger.info(f"Received RAG response of length {len(rag_response)}")
                logger.debug(f"RAG response: {rag_response[:100]}...")
                
                results.append({
                    "question": question,
                    "document_id": document['id'],
                    "document_text": document['document'],
                    "document_metadata": document['metadata'],
                    "rag_response": rag_response
                })
                
            except Exception as e:
                logger.error(f"Error processing question with RAG: {str(e)}")
        
        return results
    
    def evaluate_responses(self, processed_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate the quality of RAG responses using Anthropic API.
        
        Args:
            processed_questions: List of processed questions with RAG responses
            
        Returns:
            List of dictionaries with evaluation results
        """
        evaluations = []
        # wait for a minute to avoid rate limiting
        logger.info("Sleeping for 60 seconds to avoid rate limiting...")
        time.sleep(60)

        for item in processed_questions:
            question = item["question"]
            document_text = item["document_text"]
            rag_response = item["rag_response"]
            
            logger.info(f"Evaluating response for question: {question}")
            
            # Create a prompt for evaluation
            system_prompt = """
            You are an expert evaluator of AI assistant responses for a retrieval-augmented generation (RAG) system.
            Your job is to evaluate how well the RAG system's response answers the user's question based on the retrieved document.
            
            Provide an evaluation score from 0 to 10, where:
            - 0: Completely irrelevant, incorrect, or misleading
            - 5: Partially correct but missing key information or containing some errors
            - 10: Perfect, comprehensive, accurate, and well-explained answer
            
            Also provide a brief explanation (2-3 sentences) of your evaluation that addresses:
            1. How well the response answers the specific question
            2. Whether the response is accurate given the retrieved document
            3. If the response is satisfactory for a computer engineering student
            4. If the response is missing important information or is misleading
            
            YOUR RESPONSE MUST BE A JSON OBJECT WITH EXACTLY TWO FIELDS:
            - "score": a number from 0 to 10
            - "explanation": a string with your brief explanation

            Important:
            - The responces may refer to figures or tables, do not consider them in your evaluation and don't base your question on them.
            
            Example: {"score": 7, "explanation": "The response addresses the main question accurately but omits some details about X that would be useful for a complete understanding. The information provided is correct based on the source document, and would be helpful but not comprehensive for a computer engineering student."}
            """
            
            user_prompt = f"""
            QUESTION: {question}
            
            RETRIEVED DOCUMENT:
            {document_text}
            
            RAG SYSTEM RESPONSE:
            {rag_response}
            
            Please evaluate the quality of the RAG system's response. Return only a JSON object with the score and explanation.
            """
            
            try:
                # Generate evaluation using the LLM
                response = self.llm.generate_response(
                    context="",  # Not needed for this generation
                    query=user_prompt,
                    system_prompt=system_prompt
                )
                
                # Parse the evaluation response
                try:
                    # First try to parse the entire response as JSON
                    evaluation = json.loads(response)
                    
                    # Check if required fields are present
                    if "score" not in evaluation or "explanation" not in evaluation:
                        logger.warning("Evaluation missing required fields, attempting fallback parsing")
                        # Try to extract score and explanation manually
                        import re
                        score_match = re.search(r'"score":\s*(\d+(?:\.\d+)?)', response)
                        explanation_match = re.search(r'"explanation":\s*"([^"]+)"', response)
                        
                        if score_match and explanation_match:
                            evaluation = {
                                "score": float(score_match.group(1)),
                                "explanation": explanation_match.group(1)
                            }
                        else:
                            # Default values if parsing fails
                            evaluation = {
                                "score": 5.0,
                                "explanation": "Could not parse evaluation response"
                            }
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract evaluation manually
                    logger.warning("JSON parsing failed for evaluation, attempting manual extraction")
                    import re
                    score_match = re.search(r'score[:\s]+(\d+(?:\.\d+)?)', response, re.IGNORECASE)
                    score = float(score_match.group(1)) if score_match else 5.0
                    
                    # Try to find an explanation section
                    explanation_match = re.search(r'explanation[:\s]+(.*?)(?:\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
                    explanation = explanation_match.group(1).strip() if explanation_match else "Could not parse explanation"
                    
                    evaluation = {
                        "score": score,
                        "explanation": explanation
                    }
                
                # Ensure score is a number between 0 and 10
                try:
                    score = float(evaluation["score"])
                    score = max(0, min(10, score))  # Clamp between 0 and 10
                    evaluation["score"] = score
                except (ValueError, TypeError):
                    evaluation["score"] = 5.0
                    logger.warning(f"Could not parse score as a number: {evaluation['score']}")
                
                logger.info(f"Evaluation score: {evaluation['score']}")
                logger.info(f"Evaluation explanation: {evaluation['explanation']}")
                
                # Add evaluation to the results
                item_with_evaluation = item.copy()
                item_with_evaluation.update({
                    "evaluation_score": evaluation["score"],
                    "evaluation_explanation": evaluation["explanation"]
                })
                
                evaluations.append(item_with_evaluation)
                
            except Exception as e:
                logger.error(f"Error evaluating response: {str(e)}")
                
                # Add a default evaluation in case of error
                item_with_evaluation = item.copy()
                item_with_evaluation.update({
                    "evaluation_score": 0.0,
                    "evaluation_explanation": f"Error during evaluation: {str(e)}"
                })
                
                evaluations.append(item_with_evaluation)
        
        return evaluations
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run the complete evaluation process.
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting RAG quality evaluation")
        
        # Step 1: Select random documents
        documents = self.select_random_documents()
        if not documents:
            logger.error("No documents to evaluate")
            return {"error": "No documents available for evaluation"}
        
        logger.info(f"Selected {len(documents)} documents for evaluation")
        
        all_processed_questions = []
        
        # Step 2 & 3: Generate questions and process with RAG
        for doc in documents:
            logger.info(f"Processing document {doc['id']}")
            
            # Generate questions for this document
            questions = self.generate_questions(doc)
            
            # Process questions with RAG
            processed_questions = self.process_questions_with_rag(doc, questions)
            
            all_processed_questions.extend(processed_questions)
        
        # Step 4: Evaluate responses
        evaluations = self.evaluate_responses(all_processed_questions)
        
        # Calculate summary statistics
        if evaluations:
            avg_score = sum(item["evaluation_score"] for item in evaluations) / len(evaluations)
            score_distribution = {}
            for score_range in range(0, 11, 1):
                count = sum(1 for item in evaluations if score_range <= item["evaluation_score"] < score_range + 1)
                score_distribution[f"{score_range}"] = count
        else:
            avg_score = 0
            score_distribution = {}
        
        # Compile results
        self.results["evaluations"] = evaluations
        self.results["summary"] = {
            "average_score": avg_score,
            "total_questions": len(evaluations),
            "score_distribution": score_distribution
        }
        
        # Save results to file
        try:
            with open(self.evaluation_output_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Saved evaluation results to {self.evaluation_output_path}")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")
        
        logger.info(f"Evaluation complete. Average score: {avg_score:.2f} over {len(evaluations)} questions")
        
        return self.results


def main():
    """
    Main function to run the RAG quality evaluation.
    """
    load_dotenv()
    
    # Check if Anthropic API key is available
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not found in environment variables")
        return
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate RAG system response quality')
    parser.add_argument('--samples', type=int, default=5, help='Number of random documents to select')
    parser.add_argument('--questions', type=int, default=2, help='Number of questions per document')
    parser.add_argument('--output', type=str, default='rag_evaluation_results.json', help='Output file path for results')
    parser.add_argument('--model', type=str, default="claude-3-5-sonnet-20241022", help='Anthropic model to use')
    parser.add_argument('--collection', type=str, default="Computer_Architecture", help='ChromaDB collection name to evaluate')
    parser.add_argument('--results_per_topic', type=int, default=2, help='Number of results to retrieve per topic')
    args = parser.parse_args()
    
    logger.info(f"Starting evaluation with {args.samples} samples and {args.questions} questions per document")
    
    # Initialize LLM
    llm = AnthropicLLM(model=args.model)
    
    # Initialize and run evaluator
    evaluator = RAGQualityEvaluator(
        llm=llm,
        num_samples=args.samples,
        questions_per_document=args.questions,
        results_per_topic=args.results_per_topic,
        evaluation_output_path=args.output,
        collection_name=args.collection
    )
    
    results = evaluator.run_evaluation()
    
    # Print summary
    if "summary" in results:
        print("\n===================== EVALUATION SUMMARY =====================")
        print(f"Average Score: {results['summary']['average_score']:.2f}/10")
        print(f"Total Questions: {results['summary']['total_questions']}")
        print("\nScore Distribution:")
        for score, count in sorted(results['summary']['score_distribution'].items()):
            print(f"  Score {score}: {count} questions")
        print("==============================================================")


if __name__ == "__main__":
    main()
