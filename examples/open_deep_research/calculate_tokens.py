#!/usr/bin/env python3
"""
Token Statistics Calculator for smolagents Multi-Task Multi-Agent Logs

Calculates token usage for each task in a log file containing multiple tasks.
Supports: manager agent, search_agent, coder_agent

Usage:
    python3 calculate_tokens_correct.py [LOG_FILE]
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict


def extract_all_steps(log_content: str) -> List[Tuple[int, int, int, int]]:
    """Extract all step records with line numbers."""
    pattern = r'\[Step (\d+):.*?Input tokens: ([\d,]+).*?Output tokens: ([\d,]+)\]'
    steps = []
    
    for match in re.finditer(pattern, log_content):
        line_num = log_content[:match.start()].count('\n') + 1
        step = int(match.group(1))
        input_tok = int(match.group(2).replace(',', ''))
        output_tok = int(match.group(3).replace(',', ''))
        steps.append((line_num, step, input_tok, output_tok))
    
    return steps


def group_by_agent_using_step_sequence(steps: List[Tuple[int, int, int, int]]) -> List[List[Tuple]]:
    """
    Group steps by agent using Step number sequence.
    
    When Step number decreases or jumps backwards, it's a new agent starting.
    """
    if not steps:
        return []
    
    groups = []
    current_group = [steps[0]]
    prev_step_num = steps[0][1]
    
    for i in range(1, len(steps)):
        curr_step_num = steps[i][1]
        
        # If step number decreased or jumped back, new agent
        if curr_step_num <= prev_step_num:
            groups.append(current_group)
            current_group = [steps[i]]
        else:
            current_group.append(steps[i])
        
        prev_step_num = curr_step_num
    
    if current_group:
        groups.append(current_group)
    
    return groups


def identify_agent_types(groups: List[List[Tuple]], log_content: str) -> List[Tuple[str, List[Tuple]]]:
    """
    Identify which agent each group belongs to.
    Supports: manager, search_agent, coder_agent
    """
    identified = []
    lines = log_content.split('\n')
    
    for group in groups:
        first_line = group[0][0]
        agent_name = 'manager'  # default
        
        # Look backwards from this group's first line
        for i in range(first_line - 1, max(0, first_line - 200), -1):
            if i < len(lines):
                line_text = lines[i]
                if 'New run - search_agent' in line_text or '─ search_agent ─' in line_text:
                    agent_name = 'search_agent'
                    break
                elif 'New run - coder_agent' in line_text or '─ coder_agent ─' in line_text:
                    agent_name = 'coder_agent'
                    break
                elif 'New run' in line_text and 'search_agent' not in line_text and 'coder_agent' not in line_text:
                    agent_name = 'manager'
                    break
        
        identified.append((agent_name, group))
    
    return identified


def calculate_totals(identified_groups: List[Tuple[str, List[Tuple]]]) -> Dict[str, Dict]:
    """
    Calculate final tokens for each agent type.
    
    For managed agents that appear multiple times, take MAX.
    For manager, take LAST.
    """
    agent_data = {}
    
    for agent_name, group in identified_groups:
        last_step = group[-1]
        input_tok = last_step[2]
        output_tok = last_step[3]
        
        if agent_name not in agent_data:
            agent_data[agent_name] = {
                'inputs': [input_tok],
                'outputs': [output_tok],
                'step_counts': [len(group)]
            }
        else:
            agent_data[agent_name]['inputs'].append(input_tok)
            agent_data[agent_name]['outputs'].append(output_tok)
            agent_data[agent_name]['step_counts'].append(len(group))
    
    # Calculate finals
    results = {}
    for agent_name, data in agent_data.items():
        # All agents: take LAST appearance value
        # This represents the final cumulative state of that agent's monitor
        final_input = data['inputs'][-1]
        final_output = data['outputs'][-1]
        
        results[agent_name] = {
            'input': final_input,
            'output': final_output,
            'total': final_input + final_output,
            'appearances': len(data['inputs']),
            'total_steps': sum(data['step_counts'])
        }
    
    return results


def find_task_boundaries(log_content: str) -> List[int]:
    """Find all task start positions (marked by '──── New run ────')."""
    task_starts = []
    for match in re.finditer(r'─+\s*New run\s*─+', log_content):
        line_num = log_content[:match.start()].count('\n') + 1
        char_pos = match.start()
        task_starts.append(char_pos)
    return task_starts


def extract_debug_tokens(log_content: str, start_pos: int, end_pos: int) -> int:
    """Extract [DEBUG] 总token数: xxx from a specific section of log."""
    section = log_content[start_pos:end_pos]
    pattern = r'\[DEBUG\]\s+总token数:\s+(\d+)'
    total = 0
    for match in re.finditer(pattern, section):
        token_num = int(match.group(1))
        total += token_num
    return total


def split_steps_by_tasks(steps: List[Tuple[int, int, int, int]], 
                         task_boundaries: List[int],
                         log_content: str) -> List[List[Tuple[int, int, int, int]]]:
    """Split steps into separate tasks based on task boundaries."""
    if not task_boundaries:
        return [steps] if steps else []
    
    # Convert steps to include char position
    steps_with_pos = []
    pattern = r'\[Step (\d+):.*?Input tokens: ([\d,]+).*?Output tokens: ([\d,]+)\]'
    for match in re.finditer(pattern, log_content):
        line_num = log_content[:match.start()].count('\n') + 1
        step = int(match.group(1))
        input_tok = int(match.group(2).replace(',', ''))
        output_tok = int(match.group(3).replace(',', ''))
        char_pos = match.start()
        steps_with_pos.append((line_num, step, input_tok, output_tok, char_pos))
    
    # Group steps by task
    task_steps = []
    task_boundaries_extended = task_boundaries + [len(log_content)]
    
    for i in range(len(task_boundaries)):
        start_pos = task_boundaries[i]
        end_pos = task_boundaries_extended[i + 1]
        
        task_step_list = [(s[0], s[1], s[2], s[3]) for s in steps_with_pos 
                          if start_pos <= s[4] < end_pos]
        if task_step_list:
            task_steps.append(task_step_list)
    
    return task_steps


def main():
    # Parse arguments
    if len(sys.argv) > 1:
        log_file = Path(sys.argv[1])
    else:
        log_file = Path("output/logs/test.log")
    
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        print(f"\nUsage: python3 {sys.argv[0]} [LOG_FILE]")
        sys.exit(1)
    
    # Read log
    with open(log_file, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    # Find task boundaries
    task_boundaries = find_task_boundaries(log_content)
    num_tasks = len(task_boundaries)
    
    if num_tasks == 0:
        print("No tasks found in log file.")
        sys.exit(1)
    
    # Process each task
    all_steps = extract_all_steps(log_content)
    task_step_lists = split_steps_by_tasks(all_steps, task_boundaries, log_content)
    
    print("=" * 80)
    print(f"Token Statistics - Total Tasks: {num_tasks}")
    print("=" * 80)
    print()
    
    grand_total_input = 0
    grand_total_output = 0
    task_boundaries_extended = task_boundaries + [len(log_content)]
    
    for task_num, task_steps in enumerate(task_step_lists, 1):
        if not task_steps:
            continue
        
        # Process this task
        groups = group_by_agent_using_step_sequence(task_steps)
        identified = identify_agent_types(groups, log_content)
        totals = calculate_totals(identified)
        
        # Calculate task total from step tokens
        task_input = sum(data['input'] for data in totals.values())
        task_output = sum(data['output'] for data in totals.values())
        
        # Extract DEBUG tokens for this task and add to input
        start_pos = task_boundaries[task_num - 1]
        end_pos = task_boundaries_extended[task_num]
        debug_tokens = extract_debug_tokens(log_content, start_pos, end_pos)
        
        # Add DEBUG tokens to input
        task_input += debug_tokens
        task_total = task_input + task_output
        
        grand_total_input += task_input
        grand_total_output += task_output
        
        print(f"Task {task_num}:")
        print(f"  Input tokens:  {task_input:>12,}")
        print(f"  Output tokens: {task_output:>12,}")
        print(f"  Total tokens:  {task_total:>12,}")
        print()
    
    grand_total = grand_total_input + grand_total_output
    avg_tokens = grand_total / num_tasks if num_tasks > 0 else 0
    
    print("=" * 80)
    print("Summary:")
    print(f"  Total tasks:         {num_tasks:>10}")
    print(f"  Total input tokens:  {grand_total_input:>12,}")
    print(f"  Total output tokens: {grand_total_output:>12,}")
    print(f"  Total tokens:        {grand_total:>12,}")
    print(f"  Average tokens:      {avg_tokens:>12,.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
