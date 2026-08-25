#!/usr/bin/env python3
"""
BrowseComp-ZH Evaluator - Evaluate agent responses on BrowseComp-ZH dataset
Usage: python browsecomp-zh_evaluator.py <results_file.jsonl>
"""

import argparse
import json
import os
import re
import copy
import concurrent.futures
from typing import Dict, List, Any
from dotenv import load_dotenv
from tqdm import tqdm
from smolagents import OpenAIServerModel

load_dotenv()

# Chinese grading template based on the provided JUDGE_PROMPT_CN
JUDGE_PROMPT_CN = """根据以下精确且明确的[response]，判断以下对[question]的[correct_answer]是否正确。

[question]:  {question}

[response]:  {response}

您的判断必须符合以下指定的格式和标准：

extracted_final_answer: 从[response]中提取的最终准确答案。如果无法从答案中提取出准确的最终答案，则将提取的答案填写为"None"。

[correct_answer]: {correct_answer}

reasoning: 根据[correct_answer]解释提取的最终答案正确或错误的原因， 仅关注[correct_answer]和提取的最终答案之间是否存在有意义的差异。请勿评论问题的任何背景，请勿尝试解决问题，请勿争论任何与[correct_answer]不同的答案，仅关注答案是否匹配。

correct: 如果提取的最终答案与上面给出的[correct_answer]相符，或者在数值问题的误差范围内，则回答"yes"。否则，例如，如果存在任何不一致、歧义、不等同，或者提取的答案不正确，则回答"no"。

confidence: 从[response]中提取的置信度分数，介于0% 到100% 之间。如果没有可用的置信度分数，则填写100%。

"""


class BrowseCompZHEvaluator:
    def __init__(self, grader_model_id: str = "gpt-4", max_workers: int = 8):
        """Initialize the evaluator with a grader model."""
        self.grader_model = OpenAIServerModel(
            model_id=grader_model_id,
            api_base=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0,
            max_tokens=2048
        )
        self.max_workers = max_workers
    
    def get_remote_response(self, message, is_judge=True):
        """Get response from the grader model."""
        try:
            response_obj = self.grader_model(message)
            response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
            return response
        except Exception as e:
            print(f"Error getting response: {e}")
            return ""
    
    def grade_sample(self, question: str, correct_answer: str, response: str) -> Dict[str, Any]:
        """Grade a single sample using the grader model."""
        message = [
            {"role": "system", "content": "you are a helpful assistant!"},
            {"role": "user", "content": JUDGE_PROMPT_CN.format(
                question=question,
                response=response or "No response provided",
                correct_answer=correct_answer
            )},
        ]
        
        try:
            grading_response = self.get_remote_response(message, True)
            
            # Extract grading result using regex pattern
            pattern = r"""
                \*{0,2}extracted_final_answer\*{0,2}\s*?:\s*(.*?)\n
                \*{0,2}reasoning\*{0,2}\s*:\s*?(.*?)\n
                \*{0,2}correct\*{0,2}\s*:\s*?(.*?)\n
                \*{0,2}confidence\*{0,2}\s*?:\s*(.*?)$
                """
            matches = re.search(pattern, grading_response, re.DOTALL | re.VERBOSE)

            if matches:
                model_extracted_answer = matches.group(1).strip()
                reasoning = matches.group(2).strip()
                is_correct = matches.group(3).strip()
                model_extracted_confidence = matches.group(4).strip()
            else:
                model_extracted_answer, reasoning, is_correct, model_extracted_confidence = "", "", "", ""
            
            # Convert is_correct to boolean
            is_correct_bool = is_correct.lower() == "yes" if is_correct else False
            
            return {
                "is_correct": is_correct_bool,
                "extracted_answer": model_extracted_answer,
                "reasoning": reasoning,
                "confidence": model_extracted_confidence,
                "grading_response": grading_response
            }
            
        except Exception as e:
            print(f"Error grading sample: {e}")
            return {
                "is_correct": False,
                "extracted_answer": "Error",
                "reasoning": f"Grading error: {str(e)}",
                "confidence": "0%",
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
    
    def generate_infer_eval(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate evaluation for inference results using concurrent processing."""
        print(f"Evaluating {len(results)} results...")
        
        # Prepare messages for grading
        messages = []
        for i, result in enumerate(results):
            question = result.get('question', '')
            prediction = result.get('prediction', '')
            true_answer = result.get('true_answer', '')
            task_id = result.get('task_id', '')
            
            # Handle null or empty predictions
            if prediction == "null" or prediction is None or prediction == "":
                prediction = "No response provided"
            
            # Debug print for first few samples
            print(f"  Task ID: {task_id}")
            print(f"  Prediction: {prediction}")
            print(f"  True Answer: {true_answer}")
            print()
            
            message = [
                {"role": "system", "content": "you are a helpful assistant!"},
                {"role": "user", "content": JUDGE_PROMPT_CN.format(
                    question=question,
                    response=prediction,
                    correct_answer=true_answer
                )},
            ]
            messages.append(message)
        
        evaluation_results = []
        
        # # Process with concurrent futures
        # with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        #     futures = [executor.submit(self.get_remote_response, message, True) for message in messages]
            
        #     for i, future in tqdm(enumerate(futures), desc="Generating infer eval responses", total=len(futures)):
        #         response = future.result()
                
        #         # Debug print for first sample
        #         if i == 0:
        #             print(f"First grading response:\n{response}\n")
                
        #         # Parse the response using regex
        #         pattern = r"""
        #             \*{0,2}extracted_final_answer\*{0,2}\s*?:\s*(.*?)\n
        #             \*{0,2}reasoning\*{0,2}\s*:\s*?(.*?)\n
        #             \*{0,2}correct\*{0,2}\s*:\s*?(.*?)\n
        #             \*{0,2}confidence\*{0,2}\s*?:\s*(.*?)$
        #             """
        #         matches = re.search(pattern, response, re.DOTALL | re.VERBOSE)

        #         if matches:
        #             model_extracted_answer = matches.group(1).strip()
        #             reasoning = matches.group(2).strip()
        #             is_correct = matches.group(3).strip()
        #             model_extracted_confidence = matches.group(4).strip()
        #         else:
        #             # Try alternative parsing without asterisks
        #             alt_pattern = r"extracted_final_answer:\s*(.*?)\nreasoning:\s*(.*?)\ncorrect:\s*(.*?)\nconfidence:\s*(.*?)(?:\n|$)"
        #             alt_matches = re.search(alt_pattern, response, re.DOTALL)
        #             if alt_matches:
        #                 model_extracted_answer = alt_matches.group(1).strip()
        #                 reasoning = alt_matches.group(2).strip()
        #                 is_correct = alt_matches.group(3).strip()
        #                 model_extracted_confidence = alt_matches.group(4).strip()
        #             else:
        #                 model_extracted_answer, reasoning, is_correct, model_extracted_confidence = "", "", "", ""
        #                 if i == 0:
        #                     print(f"Failed to parse response. Raw response:\n{response}")
                
        #         # Convert is_correct to boolean
        #         is_correct_bool = is_correct.lower() == "yes" if is_correct else False
                
        #         # Handle null predictions
        #         original_prediction = results[i].get("prediction", "")
        #         if original_prediction == "null" or original_prediction is None:
        #             original_prediction = "No response provided"
                
        #         eval_result = {
        #             "task_id": results[i].get("task_id", f"unknown_{i}"),
        #             "topic": results[i].get("topic", "unknown"),
        #             "question": results[i].get("question", ""),
        #             "true_answer": results[i].get("true_answer", ""),
        #             "prediction": original_prediction,
        #             "is_correct": is_correct_bool,
        #             "extracted_answer": model_extracted_answer,
        #             "reasoning": reasoning,
        #             "confidence": model_extracted_confidence,
        #             "grading_response": response
        #         }
        #         evaluation_results.append(eval_result)
        
        return evaluation_results
    
    def evaluate(self, results_file: str) -> Dict[str, Any]:
        """Evaluate all results in the file."""
        results = self.load_results(results_file)
        
        if not results:
            print("No results to evaluate")
            return {"accuracy": 0.0, "total": 0, "correct": 0, "details": []}
        
        # Generate evaluation results
        evaluation_results = self.generate_infer_eval(results)
        
        # Calculate statistics
        correct_count = sum(1 for result in evaluation_results if result["is_correct"])
        total_count = len(evaluation_results)
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
        print("BROWSECOMP-ZH EVALUATION RESULTS")
        print("="*60)
        print(f"Total questions: {total}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Topic-wise breakdown
        if details:
            topic_stats = {}
            for result in details:
                topic = result["topic"]
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
    
    def save_detailed_results(self, evaluation_results: Dict[str, Any], output_file: str):
        """Save detailed evaluation results to file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
            print(f"\nDetailed results saved to: {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BrowseComp-ZH results")
    parser.add_argument("results_file", help="Path to the results JSONL file")
    parser.add_argument("--grader-model", default="gpt-4", 
                        help="Model to use for grading (default: gpt-4)")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Maximum number of concurrent workers (default: 8)")
    parser.add_argument("--output", help="Output file to save detailed evaluation results")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = BrowseCompZHEvaluator(
        grader_model_id=args.grader_model, 
        max_workers=args.max_workers
    )
    
    # Run evaluation
    evaluation_results = evaluator.evaluate(args.results_file)
    
    # Print summary
    evaluator.print_summary(evaluation_results)
    
    # Save detailed results if requested
    if args.output:
        evaluator.save_detailed_results(evaluation_results, args.output)


if __name__ == "__main__":
    main()
