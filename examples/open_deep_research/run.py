import argparse
import os
import threading

from dotenv import load_dotenv
from huggingface_hub import login
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

from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    # HfApiModel,
    OpenAIServerModel,
    #TransformersModel,
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
    parser.add_argument(
        "question", type=str, help="for example: 'How many studio albums did Mercedes Sosa release before 2007?'"
    )
    parser.add_argument("--model-id", type=str, default="o1")
    parser.add_argument("--enable-task-decomposition", action="store_true", 
                        help="Enable initial task decomposition, default is disabled")
    return parser.parse_args()


custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


def create_agent(model_id="o1", enable_task_decomposition=False):
    model_params = {
        "model_id": model_id,
        "api_base": os.getenv("OPENAI_BASE_URL"), #openai
        "api_key":os.getenv("OPENAI_API_KEY"),
        "custom_role_conversions": custom_role_conversions,
        "temperature": 0.1,
        "max_completion_tokens": 4096,
    }
    if model_id == "o1":
        model_params["reasoning_effort"] = "high"
    model = OpenAIServerModel(**model_params)

    # Create the rethink model using DeepSeek-R1
    rethink_model_params = {
        "model_id": "DMXAPI-HuoShan-DeepSeek-R1-671B-64k",
        "api_base": os.getenv("RETHINK_BASE_URL"),
        "api_key": os.getenv("RETHINK_API_KEY"),
        "custom_role_conversions": custom_role_conversions,
        "temperature": 0.1,
    }
    rethink_model = OpenAIServerModel(**rethink_model_params)

    text_limit = 100000
    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    WEB_TOOLS = [
        GoogleSearchTool(provider="serpapi"),
        # GoogleSearchTool(provider="serper"),
        VisitTool(browser),
        FetchHtmlTool(browser),
        FetchRawHtmlTool(browser),
        FetchPdfTool(browser),
        ArchiveSearchTool(browser),
        WikipediaHistoryTool(browser),
        TextInspectorTool(model, text_limit),
    ]

    web_search_model_params = {
        "model_id": os.getenv("RETRIEVE_MODEL"),
        "api_base": os.getenv("RETRIEVE_BASE_URL"),
        "api_key": os.getenv("RETRIEVE_API_KEY"),
        "custom_role_conversions": custom_role_conversions,
        "temperature": 0.1,
    }

    web_search_model = OpenAIServerModel(**web_search_model_params)

    text_webbrowser_agent = ToolCallingAgent(
        model=web_search_model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=1,
        planning_interval=6,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
        rethink_model=rethink_model,
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
        "model_id": "google/gemini-2.5-pro",
        "api_base": os.getenv("CODER_BASE_URL"),
        "api_key": os.getenv("CODER_API_KEY"),
        "custom_role_conversions": custom_role_conversions,
        "temperature": 0.1,
    }
    coder_model = OpenAIServerModel(**coder_model_params)
    coder_agent = SimpleCoder(
        model=coder_model,
        tools=[ti_tool],
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        name="coder_agent",
        description="""A team member specialized in algorithms and Python programming tasks.
    Ask him whenever you need algorithmic or coding support.
    Give him as much information as possible, he don't know the context you have.
    Do not instruct him on how to implement the solution. He will independently interpret the description and determine the best implementation strategy.""",
        provide_run_summary=True,
    )

    # Create ValidationAgent model
    validation_model_params = {
        "model_id": os.getenv("VALIDATION_MODEL", "o3"),
        "api_base": os.getenv("VALIDATION_BASE_URL"),
        "api_key": os.getenv("VALIDATION_API_KEY"),
        "temperature": 0.1,
        "custom_role_conversions": custom_role_conversions,
    }
    validation_model = OpenAIServerModel(**validation_model_params)
    validation_agent = ValidationAgent(validation_model)

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, TextInspectorTool(model, text_limit), DownloadTool(browser)],
        max_steps=12,
        verbosity_level=1,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=6,
        managed_agents=[text_webbrowser_agent, coder_agent],
        rethink_model=rethink_model,
        validation_agent=validation_agent,
        enable_initial_task_decomposition=enable_task_decomposition,
    )

    return manager_agent


def main():
    args = parse_args()

    agent = create_agent(model_id=args.model_id, enable_task_decomposition=args.enable_task_decomposition)

    answer = agent.run(args.question)

    agent_memory = agent.write_memory_to_messages(summary_mode=True)

    print(f"agent_memory: {agent_memory}")
    print(f"Got this answer: {answer}")


if __name__ == "__main__":
    main()
