"""
python run_webwalker.py --concurrency 32 \
  --run-name webwalker_task_1to200 \
  --model-id deepseek-v4-pro \
  | tee -a /backup/zhangxiangxin/data/ds/logs/webwalker_task_deepseekv4_1to200.log
"""
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import datasets
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm

from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    OpenAIServerModel,
    Model,
    ToolCallingAgent,
    SimpleCoder,
    ValidationAgent,
)

# Tools used by GAIA runner, reused here
from scripts.reformulator import prepare_response_webwalker
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    SimpleTextBrowser,
    VisitTool,
    FetchHtmlTool,
    FetchRawHtmlTool,
    FetchPdfTool,
    DownloadTool,
    WikipediaHistoryTool,
)
from scripts.visual_qa import visualizer


AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
]


load_dotenv(override=True)
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    try:
        login(hf_token)
    except Exception:
        pass


append_answer_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--model-id", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--enable-task-decomposition", action="store_true",
                        help="Enable initial task decomposition, default is disabled")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Optional cap on number of questions to run")
    return parser.parse_args()


custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}


def create_agent_team(model: Model, enable_task_decomposition: bool = False) -> CodeAgent:
    # Rethink model (optional)
    rethink_model_params = {
        "model_id": os.getenv("RETHINK_MODEL"),
        "api_base": os.getenv("RETHINK_BASE_URL"),
        "api_key": os.getenv("RETHINK_API_KEY"),
        "custom_role_conversions": custom_role_conversions,
        "temperature": 0.0,
    }
    rethink_model = OpenAIServerModel(**rethink_model_params)

    text_limit = 100000
    ti_tool = TextInspectorTool(model, text_limit)

    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
    )
    browser_config = {
        "downloads_folder": "downloads_folder",
        "request_kwargs": {
            "headers": {"User-Agent": user_agent},
            "timeout": 300,
        },
    }
    os.makedirs(f"./{browser_config['downloads_folder']}", exist_ok=True)
    browser = SimpleTextBrowser(**browser_config)

    WEB_TOOLS = [
        GoogleSearchTool(provider="serper"),
        VisitTool(browser),
        FetchHtmlTool(browser),
        FetchRawHtmlTool(browser),
        FetchPdfTool(browser),
        visualizer,
        ArchiveSearchTool(browser),
        WikipediaHistoryTool(browser),
        TextInspectorTool(model, text_limit),
    ]

    web_search_model_params = {
        "model_id": os.getenv("RETRIEVE_MODEL"),
        "api_base": os.getenv("RETRIEVE_BASE_URL"),
        "api_key": os.getenv("RETRIEVE_API_KEY"),
        "custom_role_conversions": custom_role_conversions,
        "temperature": 0.0,
    }
    web_search_model = OpenAIServerModel(**web_search_model_params)

    text_webbrowser_agent = ToolCallingAgent(
        model=web_search_model,
        tools=WEB_TOOLS,
        max_steps=18,
        verbosity_level=2,
        planning_interval=6,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible (including the site root URL if available), in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Note: Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
        # rethink_model=rethink_model,  # 暂时不使用rethink_model
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    For HTML files, use the 'fetch_html' tool or 'fetch_raw_html' tool to extract relevant content based on your query.
    For PDF files, use the 'fetch_pdf' tool to extract relevant content based on your query.
    For Youtube videos, use 'visit_page' to get the transcript.
    For other non-html formats, use tool 'inspect_file_as_text' to inspect it.
    You do not have the capability to use APIs or web crawlers.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""


    coder_model_params = {
        "model_id": os.getenv("CODER_MODEL"),
        "api_base": os.getenv("CODER_BASE_URL"),
        "api_key": os.getenv("CODER_API_KEY"),
        "custom_role_conversions": custom_role_conversions,
        "temperature": 0.0,
    }
    coder_model = OpenAIServerModel(**coder_model_params)
    coder_agent = SimpleCoder(
        model=coder_model,
        tools=[ti_tool],
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        name="coder_agent",
        description="""A team member specialized in algorithms and Python programming tasks.
    Ask him whenever you need algorithmic or coding support.
    Ask him whenever you need to use APIs or web crawlers.
    Give him as much information as possible, he don't know the context you have.
    If there is a file included in the task and the file content is short, include the content directly in the task instead of just mentioning the file path.
    Do not instruct him on how to implement the solution. He will independently interpret the description and determine the best implementation strategy.""",
        provide_run_summary=True,
    )

    validation_model_params = {
        "model_id": os.getenv("VALIDATION_MODEL", "o3"),
        "api_base": os.getenv("VALIDATION_BASE_URL"),
        "api_key": os.getenv("VALIDATION_API_KEY"),
        "temperature": 0.0,
        "custom_role_conversions": custom_role_conversions,
    }
    validation_model = OpenAIServerModel(**validation_model_params)
    validation_agent = ValidationAgent(validation_model) if validation_model else None

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, ti_tool, DownloadTool(browser)],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=6,
        managed_agents=[text_webbrowser_agent, coder_agent],
        # rethink_model=rethink_model,
        validation_agent=validation_agent,
        enable_initial_task_decomposition=enable_task_decomposition,
    )
    manager_agent.prompt_templates["system_prompt"] += """When calling `final_answer` to output your final answer, always provide a rich, specific, and accurate response directly relevant to the task — include as much correct detail as possible without adding unrelated content."""

    return manager_agent


def append_answer(entry: Dict[str, Any], jsonl_file: str) -> None:
    jsonl_path = Path(jsonl_file)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with append_answer_lock, open(jsonl_path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
    assert os.path.exists(jsonl_path), "File not found!"


def answer_single_question(example: Dict[str, Any], model_id: str, answers_file: str,
                           enable_task_decomposition: bool = False) -> None:
    model_params: Dict[str, Any] = {
        "model_id": model_id,
        "api_base": os.getenv("OPENAI_BASE_URL"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "custom_role_conversions": custom_role_conversions,
        "temperature": 0.0,
    }
    if model_id == "o1":
        model_params["reasoning_effort"] = "high"
        model_params["max_completion_tokens"] = 8192
    elif "gemini-2.5-flash" in model_id or "gemini-2.5-pro" in model_id:
        pass
    else:
        model_params["max_tokens"] = 4096

    model = OpenAIServerModel(**model_params)

    document_inspection_tool = TextInspectorTool(model, 100000)
    agent = create_agent_team(model, enable_task_decomposition)

    # Build the augmented question. WebWalker has no attached files.
    # root_url is always a single URL string.
    root_url = example.get("root_url") or example.get("Root_Url") or ""
    original_problem = example["question"] if "question" in example else example.get("Question", "")
    agent.set_original_problem(original_problem)

    augmented_question = (
        "You have one question to answer. It is paramount that you provide a correct answer.\n"
        "Give it all you can: you have access to browsing tools to find the correct answer.\n"
        "Run verification steps if needed.\n\n"
        f"Root URL : {root_url}\n\n"
        f"Here is the task:\n{original_problem}\n\n"
        "You should gather enough information through page traversal to ultimately solve the task."
    )

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        final_result = agent.run(augmented_question)

        agent_memory = agent.write_memory_to_messages(summary_mode=True)
        final_result = prepare_response_webwalker(augmented_question, agent_memory, reformulation_model=model)
        output = str(final_result)

        for memory_step in agent.memory.steps:
            memory_step.model_input_messages = None
        intermediate_steps = agent_memory

        parsing_error = True if any(["AgentParsingError" in step for step in intermediate_steps]) else False
        iteration_limit_exceeded = True if "Agent stopped due to iteration limit or time limit." in output else False
        raised_exception = False
        exception = None
    except Exception as e:
        output = None
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    annotated_example: Dict[str, Any] = {
        "agent_name": model.model_id,
        "question": original_problem,
        "augmented_question": augmented_question,
        # Evaluator expects `pred`
        "pred": output,
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "start_time": start_time,
        "end_time": end_time,
        "root_url": root_url,
        "info": example.get("info") or example.get("Info"),
    }
    append_answer(annotated_example, answers_file)


def get_examples_to_answer(answers_file: str, eval_ds) -> List[Dict[str, Any]]:
    print(f"Loading answers from {answers_file}...")
    try:
        df = pd.read_json(answers_file, lines=True)
        done_questions = (df["question"] if "question" in df.columns 
                          else df["Question"] if "Question" in df.columns 
                          else pd.Series([])).dropna().tolist()
        print(f"Found {len(done_questions)} previous results!")
    except Exception as e:
        print("Error when loading records:", e)
        print("No usable records! ▶️ Starting new.")
        done_questions = []
    return [row for row in eval_ds if (row.get("question") or row.get("Question")) not in done_questions]


def main():
    args = parse_args()
    print(f"Starting run with arguments: {args}")

    # Load WebWalkerQA
    eval_ds = datasets.load_dataset("callanwu/WebWalkerQA", split="main")

    answers_file = f"outputs/webwalker_outputs/raw_output/{args.run_name}.jsonl"
    tasks_to_run = get_examples_to_answer(answers_file, eval_ds)[0:200]
    if args.max_questions is not None:
        tasks_to_run = tasks_to_run[: args.max_questions]

    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = [
            exe.submit(
                answer_single_question,
                example,
                args.model_id,
                answers_file,
                args.enable_task_decomposition,
            )
            for example in tasks_to_run
        ]
        for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing WebWalkerQA"):
            try:
                f.result()
            except Exception as e:
                print("Error during execution:", e)

    print("All tasks processed.")


if __name__ == "__main__":
    main()


