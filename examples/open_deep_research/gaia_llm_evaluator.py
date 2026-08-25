# modified from the repo https://github.com/RUC-NLPIR/WebThinker/blob/main/scripts/run_web_thinker.py
# python gaia_llm_evaluator.py file.jsonl
import json
import sys
import os
import argparse
import asyncio
from tqdm import tqdm
from typing import List
from collections import defaultdict
from openai import AsyncOpenAI

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

sys.path.append('./scripts')
from math_equivalence import is_equiv


async def llm_evaluate_equivalence_single(
    client: AsyncOpenAI,
    question: str,
    labeled_answer: str,
    pred_answer: str,
    model_name: str,
    semaphore: asyncio.Semaphore,
    retry_limit: int = 3,
) -> tuple:
    """Evaluate a single pair of answers using LLM"""
    
    # First, check basic equivalence
    if is_equiv(pred_answer, labeled_answer):
        return True, "Skipped (is_equiv passed)"
    
    # If basic check fails, use LLM for semantic evaluation
    prompt = f"""You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {labeled_answer}

Predicted Answer: {pred_answer}

Are these answers equivalent? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
"""

    for attempt in range(retry_limit):
        try:
            async with semaphore:
                chat_response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = chat_response.choices[0].message.content.strip()
                print(json.dumps(chat_response.to_dict(), indent=2))
                
                # Judge based on LLM response
                llm_judge = response_text.lower() == "correct" and \
                    not ("incorrect" in response_text.lower() or \
                         "wrong" in response_text.lower() or \
                         "not correct" in response_text.lower())
                
                return llm_judge, response_text
                
        except Exception as e:
            if attempt == retry_limit - 1:
                print(f"LLM evaluation error: {e}")
                return False, "Error"
            await asyncio.sleep(1 * (attempt + 1))
    
    return False, "Error"


async def llm_evaluate_equivalence_batch(
    questions: List[str],
    labeled_answers: List[str], 
    pred_answers: List[str],
    api_base_url: str = None,
    model_name: str = None,
    api_key: str = None,
    concurrent_limit: int = 10,
) -> List[tuple]:
    """
    Evaluate multiple answer pairs concurrently using LLM
    
    Args:
        questions: List of questions
        labeled_answers: List of labeled answers
        pred_answers: List of predicted answers
        api_base_url: API base URL
        model_name: Model name
        api_key: API key
        concurrent_limit: Concurrency limit
    
    Returns:
        List of evaluation results, each element is (is_correct, llm_response_text)
    """
    # Get from environment variables if not provided
    if api_base_url is None:
        api_base_url = os.getenv('OPENAI_BASE_URL')
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY', 'empty')
    if model_name is None:
        model_name = os.getenv('OPENAI_MODEL_ID', 'qwen2.5-72b-instruct')
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=api_base_url,
    )
    semaphore = asyncio.Semaphore(concurrent_limit)
    
    tasks = [
        llm_evaluate_equivalence_single(
            client=client,
            question=q,
            labeled_answer=l,
            pred_answer=p,
            model_name=model_name,
            semaphore=semaphore,
        )
        for q, l, p in zip(questions, labeled_answers, pred_answers)
    ]
    
    with tqdm(total=len(tasks), desc="LLM Evaluation Progress") as pbar:
        async def track_progress(task):
            result = await task
            pbar.update(1)
            return result
            
        tracked_tasks = [track_progress(task) for task in tasks]
        results = await asyncio.gather(*tracked_tasks)
    
    return results


def evaluate_jsonl_file(
    file_path: str,
    api_base_url: str = None,
    model_name: str = None,
    api_key: str = None,
    concurrent_limit: int = 10
):
    """
    Evaluate answer correctness in JSONL file using LLM
    
    Args:
        file_path: Path to JSONL file
        api_base_url: LLM API base URL
        model_name: LLM model name
        api_key: API key
        concurrent_limit: Concurrency limit
    
    Returns:
        List of evaluation results
    """
    results = []
    questions = []
    labeled_answers = []
    pred_answers = []
    raw_data = []
    task_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    print(f"Reading file: {file_path}")
    print("-" * 60)
    
    # Read all data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # 提取字段
                question = data.get('question', '')
                prediction = data.get('prediction', '').strip()
                true_answer = data.get('true_answer', '').strip()
                task_level = data.get('task', '')
                task_id = data.get('task_id', '')
                
                # 存储原始数据
                raw_data.append({
                    'line_number': line_num,
                    'data': data,
                    'question': question,
                    'prediction': prediction,
                    'true_answer': true_answer,
                    'task_level': task_level,
                    'task_id': task_id
                })
                
                questions.append(question)
                labeled_answers.append(true_answer)
                pred_answers.append(prediction)
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error at line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Processing error at line {line_num}: {e}")
                continue
    
    if not questions:
        print("No valid data found")
        return [], 0, 0, task_stats
    
    print(f"Read {len(questions)} records, starting LLM evaluation...")
    
    # Batch LLM evaluation
    llm_results = asyncio.run(llm_evaluate_equivalence_batch(
        questions=questions,
        labeled_answers=labeled_answers,
        pred_answers=pred_answers,
        api_base_url=api_base_url,
        model_name=model_name,
        api_key=api_key,
        concurrent_limit=concurrent_limit
    ))
    
    # Organize results
    total_questions = 0
    correct_answers = 0
    
    for raw_item, (llm_judge, llm_response) in zip(raw_data, llm_results):
        total_questions += 1
        is_correct = llm_judge
        
        if is_correct:
            correct_answers += 1
        
        # Build result record
        result = {
            'Question': raw_item['question'],
            'Output': raw_item['prediction'],
            'Pred_Answer': raw_item['prediction'],
            'answer': raw_item['true_answer'],
            'Metrics': {
                'llm_equal': int(is_correct),
                'llm_response': llm_response
            },
            'line_number': raw_item['line_number'],
            'task_level': raw_item['task_level'],
            'task_id': raw_item['task_id'],
            'is_correct': is_correct
        }
        
        results.append(result)
        
        # Statistics by task level
        if raw_item['task_level']:
            task_stats[str(raw_item['task_level'])]['total'] += 1
            if is_correct:
                task_stats[str(raw_item['task_level'])]['correct'] += 1
    
    return results, total_questions, correct_answers, task_stats


def print_results(results, total_questions, correct_answers, task_stats, show_details=True):
    """Print evaluation results"""
    print("\n" + "=" * 60)
    print("LLM Evaluation Results")
    print("=" * 60)
    
    if show_details:
        for result in results:
            status = "✓ Correct" if result['is_correct'] else "✗ Incorrect"
            task_info = f" (Task {result['task_level']})" if result['task_level'] else ""
            print(f"Line {result['line_number']}: {status}{task_info}")
            if result.get('task_id'):
                print(f"  task_id: {result['task_id']}")
            print(f"  Question: {result['Question'][:100]}...")
            print(f"  Predicted: {result['Pred_Answer']}")
            print(f"  True Answer: {result['answer']}")
            print(f"  LLM Response: {result['Metrics']['llm_response']}")
            print()
    else:
        for result in results:
            status = "✓" if result['is_correct'] else "✗"
            task_info = f" (T{result['task_level']})" if result['task_level'] else ""
            print(f"Line {result['line_number']}: {status}{task_info} - LLM: {result['Metrics']['llm_response']}")
    
    print("=" * 60)
    print("Overall Statistics")
    print("=" * 60)
    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {correct_answers}")
    print(f"Incorrect Answers: {total_questions - correct_answers}")
    print(f"Overall Accuracy: {(correct_answers / total_questions * 100):.2f}%" if total_questions > 0 else "No data")
    
    correct_lines = [r['line_number'] for r in results if r['is_correct']]
    if correct_lines:
        print(f"Correct answer line numbers: {', '.join(map(str, correct_lines))}")
    else:
        print("No correct answers")
    
    if task_stats:
        print("\n" + "=" * 60)
        print("Statistics by Task Level")
        print("=" * 60)
        for task_level in ['1', '2', '3']:
            if task_level in task_stats:
                stats = task_stats[task_level]
                accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                print(f"Task {task_level}:")
                print(f"  Total: {stats['total']}")
                print(f"  Correct: {stats['correct']}")
                print(f"  Incorrect: {stats['total'] - stats['correct']}")
                print(f"  Accuracy: {accuracy:.2f}%")
                print()


def save_results(output_path, results, total_questions, correct_answers, task_stats, model_name=None):
    """Save evaluation results"""
    # Get model name for prefix
    if model_name is None:
        model_name = os.getenv('OPENAI_MODEL_ID', 'qwen2.5-72b-instruct')
    
    # Generate prefix
    prefix = f"eval_by_{model_name}_"
    
    # Extract directory and filename
    output_dir = os.path.dirname(output_path)
    output_filename = os.path.basename(output_path)
    
    # Build new filenames with prefix
    if output_dir:
        output_metrics_path = os.path.join(output_dir, prefix + output_filename.replace('.jsonl', '.llm_metrics.jsonl'))
        output_overall_path = os.path.join(output_dir, prefix + output_filename.replace('.jsonl', '.llm_metrics.overall.json'))
    else:
        output_metrics_path = prefix + output_filename.replace('.jsonl', '.llm_metrics.jsonl')
        output_overall_path = prefix + output_filename.replace('.jsonl', '.llm_metrics.overall.json')
    
    # Save detailed results
    with open(output_metrics_path, 'w', encoding='utf-8') as f:
        for result in results:
            # Only save required fields
            output_item = {
                'Question': result['Question'],
                'Output': result['Output'],
                'Pred_Answer': result['Pred_Answer'],
                'answer': result['answer'],
                'Metrics': result['Metrics']
            }
            if result.get('task_id'):
                output_item['task_id'] = result['task_id']
            if result.get('task_level'):
                output_item['task_level'] = result['task_level']
            
            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
    
    # Save overall statistics
    overall_metrics = {
        'total_questions': total_questions,
        'correct_answers': correct_answers,
        'incorrect_answers': total_questions - correct_answers,
        'accuracy': (correct_answers / total_questions) if total_questions > 0 else 0,
        'task_stats': {}
    }
    
    # Add task level statistics
    for task_level, stats in task_stats.items():
        overall_metrics['task_stats'][task_level] = {
            'total': stats['total'],
            'correct': stats['correct'],
            'incorrect': stats['total'] - stats['correct'],
            'accuracy': (stats['correct'] / stats['total']) if stats['total'] > 0 else 0
        }
    
    with open(output_overall_path, 'w', encoding='utf-8') as f:
        json.dump(overall_metrics, f, indent=4, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_metrics_path}")
    print(f"Overall statistics saved to: {output_overall_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GAIA JSONL file answer correctness using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python gaia_llm_evaluator.py file.jsonl
  python gaia_llm_evaluator.py file.jsonl --brief --concurrent_limit 10
  python gaia_llm_evaluator.py file.jsonl --no-details
  
Environment variables (will be used if not provided as arguments):
  OPENAI_BASE_URL - LLM API base URL
  OPENAI_API_KEY - API key
  OPENAI_MODEL - Model name (default: qwen2.5-72b-instruct)
        """
    )
    
    parser.add_argument(
        'file', 
        help='Path to JSONL file to evaluate'
    )
    
    parser.add_argument(
        '--api_base_url',
        type=str,
        default=None,
        help='LLM API base URL (default: from OPENAI_BASE_URL env var)'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='LLM model name (default: from OPENAI_MODEL env var or qwen2.5-72b-instruct)'
    )
    
    parser.add_argument(
        '--api_key',
        type=str,
        default=None,
        help='API key (default: from OPENAI_API_KEY env var)'
    )
    
    parser.add_argument(
        '--concurrent_limit',
        type=int,
        default=32,
        help='Concurrency limit (default: 32)'
    )
    
    parser.add_argument(
        '--brief', 
        action='store_true',
        help='Brief output mode'
    )
    
    parser.add_argument(
        '--no-details', 
        action='store_true',
        help='Do not show detailed information for each question'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File does not exist - {args.file}")
        return
    
    if not args.file.endswith('.jsonl'):
        print(f"Warning: {args.file} is not a .jsonl file, continuing anyway...")
    
    try:
        # Get model name (same logic as in evaluate_jsonl_file)
        model_name = args.model_name
        if model_name is None:
            model_name = os.getenv('OPENAI_MODEL_ID', 'qwen2.5-72b-instruct')
        
        results, total_questions, correct_answers, task_stats = evaluate_jsonl_file(
            file_path=args.file,
            api_base_url=args.api_base_url,
            model_name=args.model_name,
            api_key=args.api_key,
            concurrent_limit=args.concurrent_limit
        )
        
        show_details = not args.no_details and not args.brief
        print_results(results, total_questions, correct_answers, task_stats, show_details)
        
        # Save results
        save_results(args.file, results, total_questions, correct_answers, task_stats, model_name)
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

