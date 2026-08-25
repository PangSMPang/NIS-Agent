# EXAMPLE COMMAND: python run_browsecomp.py --concurrency 1 --run-name browsecomp_task_1_claude-4 --model-id claude-sonnet-4-20250514 --downloads-folder /data1/zhangxiangxin/browsecomp_downloads | tee -a logs/browsecomp_logs/browsecomp_task_1_claude-4.log
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List
import pandas
import base64
import hashlib

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from scripts.reformulator import prepare_response
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
from tqdm import tqdm

from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    # HfApiModel,
    OpenAIServerModel,
    Model,
    ToolCallingAgent,
    SimpleCoder,
    ValidationAgent,
)


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
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--model-id", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--enable-task-decomposition", action="store_true", 
                        help="Enable initial task decomposition, default is disabled")
    parser.add_argument("--topic-filter", type=str, default=None,
                        help="Filter problems by topic (e.g., 'TV shows & movies')")
    parser.add_argument("--downloads-folder", type=str, default="/data1/zhangxiangxin/browsecomp_downloads",
                        help="Directory for downloading files (default: /data1/zhangxiangxin/browsecomp_downloads)")
    return parser.parse_args()


### IMPORTANT: EVALUATION SWITCHES

print("Make sure you deactivated Tailscale VPN, else some URLs will be blocked!")

USE_OPEN_MODELS = False

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

### LOAD EVALUATION DATASET

def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]

def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()

def load_browsecomp_dataset(topic_filter=None):
    """Load and decrypt BrowseComp dataset from CSV."""
    try:
        # Load CSV file from URL
        df = pandas.read_csv("https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv")
        
        print(f"Loaded BrowseComp dataset with {len(df)} problems.")
        
        processed_data = []
        for index, row in df.iterrows():
            # Decrypt problem and answer
            problem = decrypt(row.get("problem", ""), row.get("canary", ""))
            answer = decrypt(row.get("answer", ""), row.get("canary", ""))
            problem_topic = row.get("problem_topic", "N/A")
            
            # Apply topic filter if specified
            if topic_filter and problem_topic != topic_filter:
                continue
            
            processed_data.append({
                "task_id": f"browsecomp_{index}",
                "question": problem,
                "true_answer": answer,
                "problem_topic": problem_topic
            })
        
        if topic_filter:
            print(f"Filtered to {len(processed_data)} problems with topic: {topic_filter}")
        
        return processed_data
            
    except Exception as e:
        print(f"Error loading BrowseComp dataset: {e}")
        return []

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"


def create_agent_team(model: Model, downloads_folder: str, enable_task_decomposition=False):
    # Create BROWSER_CONFIG with the specified downloads folder
    BROWSER_CONFIG = {
        "downloads_folder": downloads_folder,
        "request_kwargs": {
            "headers": {"User-Agent": user_agent},
            "timeout": 300,
        },
    }
    
    # Ensure downloads folder exists
    os.makedirs(downloads_folder, exist_ok=True)
    
    # Create the rethink model using DeepSeek-R1
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

    browser = SimpleTextBrowser(**BROWSER_CONFIG)

    WEB_TOOLS = [
        # GoogleSearchTool(provider="serpapi"),
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
        max_steps=20,
        verbosity_level=2,
        planning_interval=6,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Note: Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
        # rethink_model=rethink_model,  # Temporarily not using rethink_model
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    For HTML files, use the 'fetch_html' tool or 'fetch_raw_html' tool to extract relevant content based on your query.
    For PDF files, use the 'fetch_pdf' tool to extract relevant content based on your query.
    For Youtube videos, use 'visit_page' to get the transcript.
    For other non-html formats, use tool 'inspect_file_as_text' to inspect it.
    You do not have the capability to use APIs or web crawlers.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    # -----------------------------
    # Create the new Coder_Agent using SimpleCoder
    # -----------------------------
    coder_model_params = {
        # "model_id": "claude-sonnet-4-20250514",
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

    # Create ValidationAgent model
    validation_model_params = {
        "model_id": os.getenv("VALIDATION_MODEL", "o3"),
        "api_base": os.getenv("VALIDATION_BASE_URL"),
        "api_key": os.getenv("VALIDATION_API_KEY"),
        "temperature": 0.0,
        "custom_role_conversions": custom_role_conversions,
    }
    validation_model = OpenAIServerModel(**validation_model_params)
    validation_agent = ValidationAgent(validation_model)

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, ti_tool, DownloadTool(browser)],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=6,
        managed_agents=[text_webbrowser_agent, coder_agent],
        rethink_model=rethink_model,
        validation_agent=validation_agent,
        enable_initial_task_decomposition=enable_task_decomposition,
    )
    
    return manager_agent


def append_answer(entry: dict, jsonl_file: str) -> None:
    jsonl_file = Path(jsonl_file)
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with append_answer_lock, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")
    assert os.path.exists(jsonl_file), "File not found!"
    print("Answer exported to file:", jsonl_file.resolve())


def answer_single_question(example, model_id, answers_file, visual_inspection_tool, downloads_folder, enable_task_decomposition=False):
    model_params = {
        "model_id": model_id,
        "api_base": os.getenv("OPENAI_BASE_URL"), #openai
        "api_key":os.getenv("OPENAI_API_KEY"),
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

    agent = create_agent_team(model, downloads_folder, enable_task_decomposition)

    # Set original problem (no attached files for BrowseComp)
    original_problem = example["question"]
    agent.set_original_problem(original_problem)

    augmented_question = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist). Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer!
Here is the task:
""" + example["question"]

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Run agent 🚀
        final_result = agent.run(augmented_question)

        agent_memory = agent.write_memory_to_messages(summary_mode=True)

        final_result = prepare_response(augmented_question, agent_memory, reformulation_model=model)

        output = str(final_result)
        for memory_step in agent.memory.steps:
            memory_step.model_input_messages = None
        intermediate_steps = agent_memory

        # Check for parsing errors which indicate the LLM failed to follow the required format
        parsing_error = True if any(["AgentParsingError" in step for step in intermediate_steps]) else False

        # check if iteration limit exceeded
        iteration_limit_exceeded = True if "Agent stopped due to iteration limit or time limit." in output else False
        raised_exception = False

    except Exception as e:
        print("Error on ", augmented_question, e)
        output = None
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    annotated_example = {
        "agent_name": model.model_id,
        "question": example["question"],
        "augmented_question": augmented_question,
        "prediction": output,
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "start_time": start_time,
        "end_time": end_time,
        "problem_topic": example["problem_topic"],
        "task_id": example["task_id"],
        "true_answer": example["true_answer"],
    }
    append_answer(annotated_example, answers_file)


def get_examples_to_answer(answers_file, eval_data) -> List[dict]:
    print(f"Loading answers from {answers_file}...")
    try:
        done_questions = pd.read_json(answers_file, lines=True)["question"].tolist()
        print(f"Found {len(done_questions)} previous results!")
    except Exception as e:
        print("Error when loading records: ", e)
        print("No usable records! ▶️ Starting new.")
        done_questions = []
    return [example for example in eval_data if example["question"] not in done_questions]


def main():
    args = parse_args()
    print(f"Starting run with arguments: {args}")

    # Load BrowseComp dataset
    eval_data = load_browsecomp_dataset(topic_filter=args.topic_filter)
    if not eval_data:
        print("No data loaded. Exiting.")
        return

    print(f"Loaded {len(eval_data)} problems from BrowseComp dataset.")
    
    # Count problems by topic
    topic_counts = {}
    for example in eval_data:
        topic = example["problem_topic"]
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    print("Problems by topic:")
    for topic, count in topic_counts.items():
        print(f"  {topic}: {count}")

    answers_file = f"outputs/browsecomp_outputs/{args.run_name}.jsonl"

    # tasks_to_run = get_examples_to_answer(answers_file, eval_data)[9:22]
    # tasks_to_run = get_examples_to_answer(answers_file, eval_data)[74:101]
    # tasks_to_run = get_examples_to_answer(answers_file, eval_data)[102:117]
    # tasks_to_run = get_examples_to_answer(answers_file, eval_data)[125:136]
    tasks_to_run = get_examples_to_answer(answers_file, eval_data)[180:190]
    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = [
            exe.submit(answer_single_question, example, args.model_id, answers_file, visualizer, args.downloads_folder, args.enable_task_decomposition)
            for example in tasks_to_run
        ]
        for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"):
            f.result()

    print("All tasks processed.")


if __name__ == "__main__":
    main()
