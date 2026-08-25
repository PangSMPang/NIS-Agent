"""
Usage:
    python webwalker_evaluator.py <input_file.jsonl> [--model gpt-4o]

python webwalker_llm_evaluator.py --model gpt-4o outputs/.../xxx.jsonl 

python webwalker_llm_evaluator.py --model qwen2.5-72b-instruct outputs/.../xxx.jsonl

"""

import os
import json
import time
import asyncio
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from datasets import load_dataset
from openai import AsyncOpenAI

# Dictionary to store questions, answers, and additional information
info_adic: Dict[str, List[Any]] = {}


def _normalize_case(value: Optional[str]) -> Optional[str]:
    return value.lower() if isinstance(value, str) else value


def _extract_fields(ex: Dict[str, Any]) -> Dict[str, Any]:
    # Dataset may use either PascalCase or lowercase keys
    question = ex.get("question") or ex.get("Question")
    answer = ex.get("answer") or ex.get("Answer")
    info = ex.get("info") or ex.get("Info") or {}
    return {"question": question, "answer": answer, "info": info}


def _load_info_cache() -> None:
    ds = load_dataset("callanwu/WebWalkerQA", split="main")
    for ex in ds:
        fields = _extract_fields(ex)
        q = fields["question"]
        if q is None:
            continue
        info_adic[q] = [fields["answer"], fields["info"]]


async def evaluate_single(
    client: AsyncOpenAI,
    question: str,
    labeled_answer: str,
    pred_answer: str,
    model_name: str,
    semaphore: asyncio.Semaphore,
    retry_limit: int = 10,
) -> int:
    """使用 LLM 评估单个答案对"""
    
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
                    temperature=0,
                )
                response_text = chat_response.choices[0].message.content.strip()
                
                print("pred:")
                print(pred_answer)
                print("question:")
                print(question)
                print("answer:")
                print(labeled_answer)
                print("LLM response:")
                print(response_text)
                print("--------------------------------")
                
                # 判断是否正确
                llm_judge = response_text.lower() == "correct" and \
                    not ("incorrect" in response_text.lower() or \
                         "wrong" in response_text.lower() or \
                         "not correct" in response_text.lower())
                
                return 1 if llm_judge else 0
                
        except Exception as e:
            print(f"LLM evaluation error (attempt {attempt + 1}/{retry_limit}): {e}")
            if attempt == retry_limit - 1:
                raise e
            await asyncio.sleep(1 * (2 ** attempt))
    
    return 0


async def eval_result_async(input_path: str, model: str = "gpt-4o") -> None:
    """
    Evaluates prediction results against reference answers and generates a report.

    Parameters:
        input_path (str): Path to the input predictions file.
        model (str): LLM model name for evaluation. Default is "gpt-4o".
    """

    model_name = model
    
    # 创建 AsyncOpenAI 客户端
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 构造输出目录和文件名
    input_dir = os.path.dirname(input_path) or "."
    input_filename = os.path.basename(input_path)
    input_stem, input_ext = os.path.splitext(input_filename)

    # 以输入文件名（去掉扩展名）作为文件夹名
    output_dir = os.path.join(input_dir, input_stem)
    os.makedirs(output_dir, exist_ok=True)

    # 主输出文件：eval_by_{model}_{input_file.jsonl}
    output_filename = f"eval_by_{model_name}_{input_filename}"
    output_path = os.path.join(output_dir, output_filename)

    data_list: List[Dict[str, Any]] = []
    visited: List[str] = []

    # Ensure output file exists
    if not os.path.exists(output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("")

    # Load already processed questions
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                visited.append(json.loads(line)["question"])
            except Exception:
                continue

    # Load and filter data
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("question") not in visited:
                # our runner uses `pred`; fall back to `prediction` if needed
                if "pred" not in data and "prediction" in data:
                    data["pred"] = data["prediction"]
                data["answer"] = info_adic.get(data.get("question"), [None, None])[0]
                if data["answer"] is not None and data.get("pred") is not None:
                    data_list.append(data)

    # 使用异步并发评估
    semaphore = asyncio.Semaphore(32)  # 限制并发数为32
    s = 0
    cnt = 0

    # 创建评估任务（带索引）
    tasks_with_idx = [
        (
            idx,
            evaluate_single(
                client,
                d["question"],
                d["answer"],
                d["pred"],
                model_name,
                semaphore
            ),
            d
        )
        for idx, d in enumerate(data_list)
    ]

    # 执行所有任务并追踪进度
    with tqdm(total=len(data_list)) as pbar:
        # 收集所有结果，保持顺序
        results = []
        for idx, task, d in tasks_with_idx:
            try:
                score = await task
                results.append((idx, d, score))
            except Exception as e:
                print(f"Error processing data at index {idx}: {e}")
                results.append((idx, d, 0))
            pbar.update(1)
        
        # 按顺序写入结果
        for idx, d, score in sorted(results, key=lambda x: x[0]):
            d["score"] = score
            cnt += score
            s += 1

            # Only keep essential fields: question, pred, answer, score
            simplified_d = {
                "question": d.get("question"),
                "pred": d.get("pred") or d.get("prediction"),
                "answer": d.get("answer"),
                "score": d["score"],
            }

            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(simplified_d, ensure_ascii=False) + "\n")

            print("Current accuracy:", cnt / s)

    # Prepare statistics for the report
    single_source_easy: List[float] = []
    single_source_medium: List[float] = []
    single_source_hard: List[float] = []
    multi_source_easy: List[float] = []
    multi_source_medium: List[float] = []
    multi_source_hard: List[float] = []
    overall: List[float] = []

    datas: List[Dict[str, Any]] = []

    # Reload processed data
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            q = item.get("question")
            if q in info_adic:
                item["info"] = info_adic[q][1]
                datas.append(item)

    for temp in datas:
        score = temp.get("score")
        if score is not None:
            info = temp.get("info", {}) or {}
            # Normalize keys and values
            q_type = info.get("type") or info.get("Type") or info.get("Hop")
            q_type = _normalize_case(q_type)
            if q_type in ("single-source", "single_source"):
                q_type = "single_source"
            elif q_type in ("multi-source", "multi_source"):
                q_type = "multi_source"

            difficulty = (
                info.get("difficulty_level")
                or info.get("Difficulty_Level")
                or info.get("Level")
            )
            difficulty = _normalize_case(difficulty)

            if q_type == "single_source":
                if difficulty == "easy":
                    single_source_easy.append(score)
                elif difficulty == "medium":
                    single_source_medium.append(score)
                elif difficulty == "hard":
                    single_source_hard.append(score)

            elif q_type == "multi_source":
                if difficulty == "easy":
                    multi_source_easy.append(score)
                elif difficulty == "medium":
                    multi_source_medium.append(score)
                elif difficulty == "hard":
                    multi_source_hard.append(score)

            overall.append(score)

    def safe_average(scores: List[float]) -> Optional[float]:
        return sum(scores) / len(scores) if scores else None

    result = {
        "single_source_easy": safe_average(single_source_easy),
        "single_source_medium": safe_average(single_source_medium),
        "single_source_hard": safe_average(single_source_hard),
        "multi_source_easy": safe_average(multi_source_easy),
        "multi_source_medium": safe_average(multi_source_medium),
        "multi_source_hard": safe_average(multi_source_hard),
        "overall": safe_average(overall),
    }

    # 报告文件：eval_by_{model}_{input_stem}_report.json
    report_filename = f"eval_by_{model_name}_{input_stem}_report.json"
    report_path = os.path.join(output_dir, report_filename)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


def eval_result(input_path: str, model: str = "gpt-4o") -> None:
    """同步包装器，用于向后兼容"""
    asyncio.run(eval_result_async(input_path, model))


if __name__ == "__main__":
    import argparse

    _load_info_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        type=str,
        help="Input prediction result path (.jsonl)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Evaluation model name (default: gpt-4o)",
    )
    args = parser.parse_args()

    eval_result(args.input_path, model=args.model)
