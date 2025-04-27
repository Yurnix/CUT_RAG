#!/usr/bin/env python3
"""
RAG Quality Evaluation Visualizer

This script visualizes the results from the RAG quality evaluation,
creating charts and summary statistics for easy analysis.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import os


def load_evaluation_results(file_path: str) -> Dict[str, Any]:
    """
    Load evaluation results from a JSON file.
    
    Args:
        file_path: Path to the evaluation results JSON file
        
    Returns:
        Dictionary containing the evaluation results
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading evaluation results: {str(e)}")
        return {}


def create_score_histogram(evaluations: List[Dict[str, Any]], output_path: str = None) -> None:
    """
    Create a histogram of evaluation scores.
    
    Args:
        evaluations: List of evaluation result dictionaries
        output_path: Optional path to save the figure
    """
    scores = [item.get("evaluation_score", 0) for item in evaluations]
    
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=11, range=(0, 11), alpha=0.7, color='royalblue', edgecolor='black')
    
    plt.title('Distribution of RAG Response Quality Scores', fontsize=15)
    plt.xlabel('Score (0-10)', fontsize=12)
    plt.ylabel('Number of Responses', fontsize=12)
    plt.xticks(range(0, 11))
    plt.grid(axis='y', alpha=0.3)
    
    # Add mean score line
    mean_score = np.mean(scores)
    plt.axvline(mean_score, color='red', linestyle='dashed', linewidth=1)
    plt.text(mean_score + 0.1, plt.ylim()[1]*0.9, f'Mean: {mean_score:.2f}', color='red')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Score histogram saved to {output_path}")
    else:
        plt.show()


def create_score_breakdown_chart(evaluations: List[Dict[str, Any]], output_path: str = None) -> None:
    """
    Create a categorical breakdown of scores.
    
    Args:
        evaluations: List of evaluation result dictionaries
        output_path: Optional path to save the figure
    """
    # Define score categories
    score_categories = {
        "Poor (0-3)": (0, 3),
        "Moderate (4-6)": (4, 6),
        "Good (7-8)": (7, 8),
        "Excellent (9-10)": (9, 10)
    }
    
    # Count evaluations in each category
    category_counts = {category: 0 for category in score_categories}
    
    for item in evaluations:
        score = item.get("evaluation_score", 0)
        for category, (min_score, max_score) in score_categories.items():
            if min_score <= score <= max_score:
                category_counts[category] += 1
                break
    
    # Create chart
    plt.figure(figsize=(10, 6))
    
    categories = list(category_counts.keys())
    counts = list(category_counts.values())
    
    bars = plt.bar(categories, counts, color=['#ff9999', '#ffcc99', '#99ccff', '#99ff99'])
    
    # Add percentage labels
    total = sum(counts)
    for bar, count in zip(bars, counts):
        percentage = (count / total) * 100 if total > 0 else 0
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
    
    plt.title('RAG Response Quality Categories', fontsize=15)
    plt.ylabel('Number of Responses', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Score breakdown chart saved to {output_path}")
    else:
        plt.show()


def create_common_issues_chart(evaluations: List[Dict[str, Any]], output_path: str = None) -> None:
    """
    Analyze explanations for common issues and create a chart.
    
    Args:
        evaluations: List of evaluation result dictionaries
        output_path: Optional path to save the figure
    """
    # Define common issue keywords to look for
    common_issues = {
        "Missing information": ["missing", "lack", "incomplete", "omit"],
        "Inaccurate content": ["incorrect", "inaccurate", "error", "wrong", "mistake"],
        "Poor explanation": ["unclear", "confusing", "poorly explained", "ambiguous"],
        "Irrelevant content": ["irrelevant", "unrelated", "not related", "off-topic"],
        "Misleading information": ["misleading", "misinterpret", "misrepresent"]
    }
    
    # Count occurrences of each issue
    issue_counts = {issue: 0 for issue in common_issues}
    
    for item in evaluations:
        explanation = item.get("evaluation_explanation", "").lower()
        
        for issue, keywords in common_issues.items():
            if any(keyword in explanation for keyword in keywords):
                issue_counts[issue] += 1
    
    # Sort issues by frequency
    sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
    issues = [x[0] for x in sorted_issues]
    counts = [x[1] for x in sorted_issues]
    
    # Create chart
    plt.figure(figsize=(10, 6))
    
    bars = plt.barh(issues, counts, color='skyblue')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        plt.text(count + 0.1, bar.get_y() + bar.get_height()/2, str(count), va='center')
    
    plt.title('Common Issues in RAG Responses', fontsize=15)
    plt.xlabel('Number of Occurrences', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Common issues chart saved to {output_path}")
    else:
        plt.show()


def create_low_high_score_examples(evaluations: List[Dict[str, Any]], output_path: str = None) -> None:
    """
    Create a report with examples of lowest and highest scoring responses.
    
    Args:
        evaluations: List of evaluation result dictionaries
        output_path: Optional path to save the report
    """
    if not evaluations:
        print("No evaluations to analyze")
        return
    
    # Sort evaluations by score
    sorted_evaluations = sorted(evaluations, key=lambda x: x.get("evaluation_score", 0))
    
    # Get lowest and highest scoring examples
    lowest_examples = sorted_evaluations[:3]
    highest_examples = sorted_evaluations[-3:]
    
    # Create report text
    report = "# RAG Quality Evaluation: Notable Examples\n\n"
    
    # Add lowest scoring examples
    report += "## Lowest Scoring Responses\n\n"
    for i, example in enumerate(lowest_examples, 1):
        score = example.get("evaluation_score", "N/A")
        question = example.get("question", "N/A")
        response = example.get("rag_response", "N/A")
        explanation = example.get("evaluation_explanation", "N/A")
        
        report += f"### Example {i} (Score: {score}/10)\n\n"
        report += f"**Question:** {question}\n\n"
        report += f"**RAG Response:**\n```\n{response}\n```\n\n"
        report += f"**Evaluation:** {explanation}\n\n"
        report += "---\n\n"
    
    # Add highest scoring examples
    report += "## Highest Scoring Responses\n\n"
    for i, example in enumerate(highest_examples, 1):
        score = example.get("evaluation_score", "N/A")
        question = example.get("question", "N/A")
        response = example.get("rag_response", "N/A")
        explanation = example.get("evaluation_explanation", "N/A")
        
        report += f"### Example {i} (Score: {score}/10)\n\n"
        report += f"**Question:** {question}\n\n"
        report += f"**RAG Response:**\n```\n{response}\n```\n\n"
        report += f"**Evaluation:** {explanation}\n\n"
        report += "---\n\n"
    
    # Save or print report
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Example report saved to {output_path}")
    else:
        print(report)


def create_summary_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a summary DataFrame with key metrics.
    
    Args:
        data: Evaluation results dictionary
        
    Returns:
        DataFrame with summary statistics
    """
    evaluations = data.get("evaluations", [])
    summary = data.get("summary", {})
    
    # Extract scores
    scores = [item.get("evaluation_score", 0) for item in evaluations]
    
    # Prepare summary data
    summary_data = {
        "Metric": [
            "Total Questions Evaluated",
            "Average Score",
            "Median Score",
            "Min Score",
            "Max Score",
            "Standard Deviation",
            "Poor Responses (0-3)",
            "Moderate Responses (4-6)",
            "Good Responses (7-8)",
            "Excellent Responses (9-10)"
        ],
        "Value": [
            len(scores),
            np.mean(scores) if scores else 0,
            np.median(scores) if scores else 0,
            min(scores) if scores else 0,
            max(scores) if scores else 0,
            np.std(scores) if scores else 0,
            sum(1 for score in scores if 0 <= score <= 3),
            sum(1 for score in scores if 4 <= score <= 6),
            sum(1 for score in scores if 7 <= score <= 8),
            sum(1 for score in scores if 9 <= score <= 10)
        ]
    }
    
    return pd.DataFrame(summary_data)


def main():
    parser = argparse.ArgumentParser(description="Visualize RAG quality evaluation results")
    parser.add_argument("--input", "-i", type=str, default="rag_evaluation_results.json", 
                        help="Input JSON file with evaluation results")
    parser.add_argument("--output-dir", "-o", type=str, default="evaluation_visualizations",
                        help="Directory to save visualizations")
    args = parser.parse_args()
    
    # Load evaluation results
    data = load_evaluation_results(args.input)
    
    if not data or "evaluations" not in data or not data["evaluations"]:
        print("No valid evaluation data found")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Generate visualizations
    evaluations = data["evaluations"]
    
    # Create histogram
    create_score_histogram(evaluations, os.path.join(args.output_dir, "score_histogram.png"))
    
    # Create score breakdown chart
    create_score_breakdown_chart(evaluations, os.path.join(args.output_dir, "score_breakdown.png"))
    
    # Create common issues chart
    create_common_issues_chart(evaluations, os.path.join(args.output_dir, "common_issues.png"))
    
    # Create examples report
    create_low_high_score_examples(evaluations, os.path.join(args.output_dir, "notable_examples.md"))
    
    # Create and save summary dataframe
    summary_df = create_summary_dataframe(data)
    summary_df.to_csv(os.path.join(args.output_dir, "summary_metrics.csv"), index=False)
    
    # Print summary
    print("\n=== RAG Evaluation Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nAll visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
