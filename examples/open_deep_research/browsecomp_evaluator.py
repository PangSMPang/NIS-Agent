#!/usr/bin/env python3
"""
BrowseComp Evaluator - Evaluate agent responses on BrowseComp dataset
Usage:
    python browsecomp_evaluator.py <results_file.jsonl> [--grader-model qwen2.5-72b-instruct]

python browsecomp_evaluator.py --grader-model qwen2.5-72b-instruct xxx.jsonl
"""

import argparse
import json
import os
import re
from typing import Dict, List, Any
from dotenv import load_dotenv
from smolagents import OpenAIServerModel

load_dotenv()

# Grading template adapted from loda_browsecomp.py
GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
""".strip()


class BrowseCompEvaluator:
    def __init__(self, grader_model_id: str = "qwen2.5-72b-instruct"):
        """Initialize the evaluator with a grader model."""
        self.grader_model_id = grader_model_id
        self.grader_model = OpenAIServerModel(
            model_id=grader_model_id,
            api_base=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0,
            max_tokens=2048
        )
    
    def grade_sample(self, question: str, correct_answer: str, response: str) -> Dict[str, Any]:
        """Grade a single sample using the grader model."""
        grader_prompt = GRADER_TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            response=response or "No response provided",
        )
        
        try:
            # Create messages for the grader model
            messages = [{"role": "user", "content": grader_prompt}]
            response_obj = self.grader_model(messages)
            grading_response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
            
            # Extract grading result
            match = re.search(r"correct:\s*(yes|no)", grading_response, re.IGNORECASE)
            is_correct = match and match.group(1).lower() == "yes"
            
            # Extract extracted answer
            answer_match = re.search(r"extracted_final_answer:\s*(.+?)(?:\n|$)", grading_response, re.DOTALL)
            extracted_answer = answer_match.group(1).strip() if answer_match else "None"
            
            # Extract reasoning
            reasoning_match = re.search(r"reasoning:\s*(.+?)(?:\ncorrect:|$)", grading_response, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
            
            return {
                "is_correct": is_correct,
                "extracted_answer": extracted_answer,
                "reasoning": reasoning,
                "grading_response": grading_response
            }
            
        except Exception as e:
            print(f"Error grading sample: {e}")
            return {
                "is_correct": False,
                "extracted_answer": "Error",
                "reasoning": f"Grading error: {str(e)}",
                "grading_response": ""
            }
    
    def load_results(self, results_file: str) -> List[Dict[str, Any]]:
        """Load results from JSONL file."""
        results = []
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        result = json.loads(line)
                        results.append(result)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num}: {e}")
                        continue
        except FileNotFoundError:
            print(f"Results file not found: {results_file}")
            return []
        except Exception as e:
            print(f"Error loading results: {e}")
            return []
        
        return results
    
    def evaluate(self, results_file: str) -> Dict[str, Any]:
        """Evaluate all results in the file."""
        results = self.load_results(results_file)
        
        if not results:
            print("No results to evaluate")
            return {"accuracy": 0.0, "total": 0, "correct": 0, "details": []}
        
        print(f"Evaluating {len(results)} results...")
        
        evaluation_results = []
        correct_count = 0
        total_count = 0
        
        for i, result in enumerate(results):
            # Extract required fields
            question = result.get("question", "")
            prediction = result.get("prediction", "")
            true_answer = result.get("true_answer", "")
            task_id = result.get("task_id", f"unknown_{i}")
            problem_topic = result.get("problem_topic", "unknown")
            
            if not question or not true_answer:
                print(f"Skipping result {i+1}: missing question or true_answer")
                continue
            
            # Grade the sample
            grading_result = self.grade_sample(question, true_answer, prediction)
            
            is_correct = grading_result["is_correct"]
            if is_correct:
                correct_count += 1
            total_count += 1
            
            evaluation_result = {
                "task_id": task_id,
                "problem_topic": problem_topic,
                "question": question,
                "true_answer": true_answer,
                "prediction": prediction,
                "is_correct": is_correct,
                "extracted_answer": grading_result["extracted_answer"],
                "reasoning": grading_result["reasoning"]
            }
            evaluation_results.append(evaluation_result)
            
            # Print progress
            if (i + 1) % 10 == 0 or (i + 1) == len(results):
                print(f"Processed {i + 1}/{len(results)} results...")
        
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "total": total_count,
            "correct": correct_count,
            "details": evaluation_results
        }
    
    def print_summary(self, evaluation_results: Dict[str, Any]):
        """Print evaluation summary."""
        accuracy = evaluation_results["accuracy"]
        total = evaluation_results["total"]
        correct = evaluation_results["correct"]
        details = evaluation_results["details"]
        
        print("\n" + "="*60)
        print("BROWSECOMP EVALUATION RESULTS")
        print("="*60)
        print(f"Total questions: {total}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Topic-wise breakdown
        if details:
            topic_stats = {}
            for result in details:
                topic = result["problem_topic"]
                if topic not in topic_stats:
                    topic_stats[topic] = {"total": 0, "correct": 0}
                topic_stats[topic]["total"] += 1
                if result["is_correct"]:
                    topic_stats[topic]["correct"] += 1
            
            print("\nTopic-wise breakdown:")
            print("-" * 40)
            for topic, stats in sorted(topic_stats.items()):
                topic_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
                print(f"{topic}: {stats['correct']}/{stats['total']} ({topic_accuracy:.3f})")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate BrowseComp results")
    parser.add_argument("results_file", help="Path to the results JSONL file")
    parser.add_argument(
        "--grader-model",
        default="qwen2.5-72b-instruct",
        help="Model to use for grading (default: qwen2.5-72b-instruct)"
    )
    # 原来的 --output 参数删除，输出路径固定规则生成

    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = BrowseCompEvaluator(grader_model_id=args.grader_model)
    
    # Run evaluation
    evaluation_results = evaluator.evaluate(args.results_file)
    
    # Print summary
    evaluator.print_summary(evaluation_results)
    
    results_path = args.results_file
    model_name = args.grader_model

    results_dir = os.path.dirname(results_path) or "."
    results_filename = os.path.basename(results_path)

    output_filename = f"eval_by_{model_name}_{results_filename}"
    output_path = os.path.join(results_dir, output_filename)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    main()
