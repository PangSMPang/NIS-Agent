import json
import sys
import os
import argparse
from collections import defaultdict

# 添加当前目录到Python路径以便导入gaia_scorer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入gaia_scorer中的评分函数
from scripts.gaia_scorer import question_scorer, check_close_call

def evaluate_jsonl_file(file_path):
    """
    评估JSONL文件中的答案正确性
    
    Args:
        file_path: JSONL文件路径
    
    Returns:
        评估结果统计
    """
    results = []
    total_questions = 0
    correct_answers = 0
    task_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    print(f"正在评估文件: {file_path}")
    print("-" * 60)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                prediction = data.get('prediction', '')
                true_answer = data.get('true_answer', '')
                task_level = data.get('task', '')
                task_id = data.get('task_id', '')  # 提取 task_id
                
                # 使用gaia_scorer的逻辑评估答案
                is_correct = question_scorer(prediction, true_answer)
                
                # 记录结果
                results.append({
                    'line_number': line_num,
                    'prediction': prediction,
                    'true_answer': true_answer,
                    'task_level': task_level,
                    'task_id': task_id,
                    'is_correct': is_correct
                })
                
                # 统计总体数据
                total_questions += 1
                if is_correct:
                    correct_answers += 1
                
                # 统计按task等级分类的数据
                if task_level:
                    task_stats[str(task_level)]['total'] += 1
                    if is_correct:
                        task_stats[str(task_level)]['correct'] += 1
                    
            except json.JSONDecodeError as e:
                print(f"第{line_num}行JSON解析错误: {e}")
                continue
            except Exception as e:
                print(f"第{line_num}行处理错误: {e}")
                continue
    
    return results, total_questions, correct_answers, task_stats

def print_results(results, total_questions, correct_answers, task_stats, show_details=True):
    """打印评估结果"""
    print("=" * 60)
    print("答案评估结果")
    print("=" * 60)
    
    if show_details:
        for result in results:
            status = "✓ 正确" if result['is_correct'] else "✗ 错误"
            task_info = f" (Task {result['task_level']})" if result['task_level'] else ""
            print(f"第{result['line_number']}行: {status}{task_info}")
            if result.get('task_id'):
                print(f"  task_id: {result['task_id']}")
            print(f"  预测答案: {result['prediction']}")
            print(f"  正确答案: {result['true_answer']}")
            print()
    else:
        for result in results:
            status = "✓" if result['is_correct'] else "✗"
            task_info = f" (T{result['task_level']})" if result['task_level'] else ""
            print(f"第{result['line_number']}行: {status}{task_info}")
    
    print("=" * 60)
    print("总体统计结果")
    print("=" * 60)
    print(f"总问题数: {total_questions}")
    print(f"正确答案数: {correct_answers}")
    print(f"错误答案数: {total_questions - correct_answers}")
    print(f"总体准确率: {(correct_answers / total_questions * 100):.2f}%" if total_questions > 0 else "无数据")
    
    correct_lines = [r['line_number'] for r in results if r['is_correct']]
    if correct_lines:
        print(f"正确答案的行号: {', '.join(map(str, correct_lines))}")
    else:
        print("没有正确答案")
    
    if task_stats:
        print("\n" + "=" * 60)
        print("按Task等级分类统计")
        print("=" * 60)
        for task_level in ['1', '2', '3']:
            if task_level in task_stats:
                stats = task_stats[task_level]
                accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                print(f"Task {task_level}:")
                print(f"  总题数: {stats['total']}")
                print(f"  正确数: {stats['correct']}")
                print(f"  错误数: {stats['total'] - stats['correct']}")
                print(f"  准确率: {accuracy:.2f}%")
                print()
        for task_level in sorted(task_stats.keys()):
            if task_level not in ['1', '2', '3']:
                stats = task_stats[task_level]
                accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                print(f"Task {task_level}:")
                print(f"  总题数: {stats['total']}")
                print(f"  正确数: {stats['correct']}")
                print(f"  错误数: {stats['total'] - stats['correct']}")
                print(f"  准确率: {accuracy:.2f}%")
                print()

def main():
    parser = argparse.ArgumentParser(
        description="使用 GAIA 评分器评估 JSONL 文件中的答案正确性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python answer_evaluator.py file.jsonl                    # 评估指定文件
  python answer_evaluator.py file.jsonl --brief           # 简洁输出模式
  python answer_evaluator.py file.jsonl --no-details      # 不显示详细信息
  python answer_evaluator.py *.jsonl                      # 评估多个文件
        """
    )
    
    parser.add_argument(
        'files', 
        nargs='+',
        help='要评估的 JSONL 文件路径（可以指定多个文件）'
    )
    
    parser.add_argument(
        '--brief', 
        action='store_true',
        help='简洁输出模式（只显示每行的正确/错误状态）'
    )
    
    parser.add_argument(
        '--no-details', 
        action='store_true',
        help='不显示每个问题的详细信息'
    )
    
    args = parser.parse_args()
    
    total_files = len(args.files)
    overall_correct = 0
    overall_total = 0
    overall_task_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for i, file_path in enumerate(args.files):
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 - {file_path}")
            continue
        
        if not file_path.endswith('.jsonl'):
            print(f"警告: {file_path} 不是 .jsonl 文件，继续处理...")
        
        try:
            results, total_questions, correct_answers, task_stats = evaluate_jsonl_file(file_path)
            overall_total += total_questions
            overall_correct += correct_answers
            
            for task_level, stats in task_stats.items():
                overall_task_stats[task_level]['total'] += stats['total']
                overall_task_stats[task_level]['correct'] += stats['correct']
            
            show_details = not args.no_details and not args.brief
            print_results(results, total_questions, correct_answers, task_stats, show_details)
            
            if i < total_files - 1:
                print("\n" + "=" * 80 + "\n")
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue
    
    if total_files > 1 and overall_total > 0:
        print("\n" + "=" * 80)
        print("总体统计结果")
        print("=" * 80)
        print(f"处理文件数: {total_files}")
        print(f"总问题数: {overall_total}")
        print(f"总正确数: {overall_correct}")
        print(f"总错误数: {overall_total - overall_correct}")
        print(f"总体准确率: {(overall_correct / overall_total * 100):.2f}%")
        
        if overall_task_stats:
            print("\n" + "=" * 60)
            print("总体按Task等级分类统计")
            print("=" * 60)
            for task_level in ['1', '2', '3']:
                if task_level in overall_task_stats:
                    stats = overall_task_stats[task_level]
                    accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                    print(f"Task {task_level}:")
                    print(f"  总题数: {stats['total']}")
                    print(f"  正确数: {stats['correct']}")
                    print(f"  错误数: {stats['total'] - stats['correct']}")
                    print(f"  准确率: {accuracy:.2f}%")
                    print()

if __name__ == "__main__":
    main()
