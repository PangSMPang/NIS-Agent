#!/usr/bin/env python3
# python aime_evaluator.py aime2025_results/qwen3-8b_ite_5.jsonl 
import json
import sys
from pathlib import Path

def evaluate_answers(file_path):
    """
    评估 JSONL 文件中的答案准确率
    """
    if not Path(file_path).exists():
        print(f"错误: 文件 {file_path} 不存在")
        return
    
    total_count = 0
    correct_count = 0
    results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # 提取字段（注意 answer 与题号在 original_problem 下）
                    final_answer = str(data.get('final_answer', '')).strip()
                    original_problem = data.get('original_problem', {}) or {}
                    expected_answer = str(original_problem.get('answer', '')).strip()
                    problem_id = original_problem.get('id', data.get('id', line_num - 1))
                    
                    
                    # 判断答案是否正确
                    is_correct = final_answer == expected_answer
                    
                    total_count += 1
                    if is_correct:
                        correct_count += 1
                    
                    results.append({
                        'problem_id': problem_id,
                        'final_answer': final_answer,
                        'expected_answer': expected_answer,
                        'correct': is_correct
                    })
                    
                except json.JSONDecodeError as e:
                    print(f"警告: 第 {line_num} 行 JSON 解析错误: {e}")
                    continue
    
    except FileNotFoundError:
        print(f"错误: 无法打开文件 {file_path}")
        return
    except Exception as e:
        print(f"错误: 读取文件时发生异常: {e}")
        return
    
    # 计算准确率
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    # 输出结果
    print(f"总题数: {total_count}")
    print(f"正确数: {correct_count}")
    print(f"准确率: {accuracy:.2f}%")
    print("\n详细结果:")
    print("题号\t预期答案\t实际答案\t结果")
    print("-" * 40)
    
    for result in results:
        status = "✓" if result['correct'] else "✗"
        print(f"{result['problem_id']}\t{result['expected_answer']}\t\t{result['final_answer']}\t\t{status}")

def main():
    if len(sys.argv) != 2:
        print("用法: python aime2025_evaluator.py <jsonl_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    evaluate_answers(file_path)

if __name__ == "__main__":
    main()
