#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import inspect
import json
import os
import re
import tempfile
import textwrap
import time
from collections import deque
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, TypedDict, Union

import jinja2
import yaml
import ast
from huggingface_hub import create_repo, metadata_update, snapshot_download, upload_folder
from jinja2 import StrictUndefined, Template
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from .agent_types import AgentAudio, AgentImage, AgentType, handle_agent_output_types
from .default_tools import TOOL_MAPPING, FinalAnswerTool
from .local_python_executor import BASE_BUILTIN_MODULES, LocalPythonExecutor, PythonExecutor, fix_final_answer_code
from .memory import ActionStep, AgentMemory, PlanningStep, SystemPromptStep, TaskStep, ToolCall
from .models import (
    ChatMessage,
    MessageRole,
    Model,
)
from .monitoring import (
    YELLOW_HEX,
    AgentLogger,
    LogLevel,
    Monitor,
)
from .remote_executors import DockerExecutor, E2BExecutor
from .tools import Tool
from .utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
    make_init_file,
    parse_code_blobs,
    parse_json_tool_call,
    truncate_content,
)


logger = getLogger(__name__)

# Global variable to store the original task/problem for managed agents
_GLOBAL_ORIGINAL_PROBLEM = ""


def get_variable_names(self, template: str) -> Set[str]:
    pattern = re.compile(r"\{\{([^{}]+)\}\}")
    return {match.group(1).strip() for match in pattern.finditer(template)}


def populate_template(template: str, variables: Dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")


class PlanningPromptTemplate(TypedDict):
    """
    Prompt templates for the planning step.

    Args:
        task_decomposition (`str`): Task decomposition prompt.
        update_facts_pre_messages (`str`): Update facts pre-messages prompt.
        update_facts_post_messages (`str`): Update facts post-messages prompt.
        update_plan_pre_messages (`str`): Update plan pre-messages prompt.
        update_plan_post_messages (`str`): Update plan post-messages prompt.
    """

    task_decomposition: str
    update_facts_pre_messages: str
    update_facts_post_messages: str
    update_plan_pre_messages: str
    update_plan_post_messages: str


class ManagedAgentPromptTemplate(TypedDict):
    """
    Prompt templates for the managed agent.

    Args:
        task (`str`): Task prompt.
        report (`str`): Report prompt.
    """

    task: str
    report: str


class FinalAnswerPromptTemplate(TypedDict):
    """
    Prompt templates for the final answer.

    Args:
        pre_messages (`str`): Pre-messages prompt.
        post_messages (`str`): Post-messages prompt.
    """

    pre_messages: str
    post_messages: str


class RethinkPromptTemplate(TypedDict):
    """
    Prompt templates for the rethink process.

    Args:
        reasoning (`str`): Reasoning validation prompt.
        format (`str`): Format checking prompt.
    """

    reasoning: str
    format: str


class PromptTemplates(TypedDict):
    """
    Prompt templates for the agent.

    Args:
        system_prompt (`str`): System prompt.
        planning ([`~agents.PlanningPromptTemplate`]): Planning prompt templates.
        managed_agent ([`~agents.ManagedAgentPromptTemplate`]): Managed agent prompt templates.
        final_answer ([`~agents.FinalAnswerPromptTemplate`]): Final answer prompt templates.
        rethink ([`~agents.RethinkPromptTemplate`], *optional*): Rethink prompt templates for reflection.
    """

    system_prompt: str
    planning: PlanningPromptTemplate
    managed_agent: ManagedAgentPromptTemplate
    final_answer: FinalAnswerPromptTemplate
    rethink: Optional[RethinkPromptTemplate]


EMPTY_PROMPT_TEMPLATES = PromptTemplates(
    system_prompt="",
    planning=PlanningPromptTemplate(
        task_decomposition="",
        update_facts_pre_messages="",
        update_facts_post_messages="",
        update_plan_pre_messages="",
        update_plan_post_messages="",
    ),
    managed_agent=ManagedAgentPromptTemplate(task="", report=""),
    final_answer=FinalAnswerPromptTemplate(pre_messages="", post_messages=""),
    rethink=None,
)


class MultiStepAgent:
    """
    Agent class that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of action (given by the LLM) and observation (obtained from the environment).

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        max_steps (`int`, default `20`): Maximum number of steps the agent can take to solve the task.
        tool_parser (`Callable`, *optional*): Function used to parse the tool calls from the LLM output.
        add_base_tools (`bool`, default `False`): Whether to add the base tools to the agent's tools.
        verbosity_level (`LogLevel`, default `LogLevel.INFO`): Level of verbosity of the agent's logs.
        grammar (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
        managed_agents (`list`, *optional*): Managed agents that the agent can call.
        step_callbacks (`list[Callable]`, *optional*): Callbacks that will be called at each step.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        name (`str`, *optional*): Necessary for a managed agent only - the name by which this agent can be called.
        description (`str`, *optional*): Necessary for a managed agent only - the description of this agent.
        provide_run_summary (`bool`, *optional*): Whether to provide a run summary when called as a managed agent.
        final_answer_checks (`list`, *optional*): List of Callables to run before returning a final answer for checking validity.
    """

    def __init__(
        self,
        tools: List[Tool],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        prompt_templates: Optional[PromptTemplates] = None,
        max_steps: int = 20,
        tool_parser: Optional[Callable] = None,
        add_base_tools: bool = False,
        verbosity_level: LogLevel = LogLevel.INFO,
        grammar: Optional[Dict[str, str]] = None,
        managed_agents: Optional[List] = None,
        step_callbacks: Optional[List[Callable]] = None,
        planning_interval: Optional[int] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        provide_run_summary: bool = False,
        final_answer_checks: Optional[List[Callable]] = None,
        validation_agent: Optional['ValidationAgent'] = None,
        enable_initial_task_decomposition: bool = True,
    ):
        self.agent_name = self.__class__.__name__
        self.model = model
        self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES
        self.max_steps = max_steps
        self.step_number = 0
        self.tool_parser = tool_parser or parse_json_tool_call
        self.grammar = grammar
        self.planning_interval = planning_interval
        self.enable_initial_task_decomposition = enable_initial_task_decomposition
        self.state = {}
        self.name = name
        self.description = description
        self.provide_run_summary = provide_run_summary
        self.final_answer_checks = final_answer_checks
        self.original_problem = None  # Store the original problem for managed agents
        self.validation_agent = validation_agent  # Use provided validation agent
        self.first_final_answer_called = False  # Track if final_answer has been called
        self.validation_step_counter = 1  # Counter for validation steps
        self._should_run_validation = False  # Flag to trigger validation step
        self._final_answer_for_validation = None  # Store final answer for validation

        self._setup_managed_agents(managed_agents)
        self._setup_tools(tools, add_base_tools)
        self._validate_tools_and_managed_agents(tools, managed_agents)

        self.system_prompt = self.initialize_system_prompt()
        self.input_messages = None
        self.task = None
        self.memory = AgentMemory(self.system_prompt)
        self.logger = AgentLogger(level=verbosity_level)
        self.monitor = Monitor(self.model, self.logger)
        self.step_callbacks = step_callbacks if step_callbacks is not None else []
        self.step_callbacks.append(self.monitor.update_metrics)

    def _setup_managed_agents(self, managed_agents):
        self.managed_agents = {}
        if managed_agents:
            assert all(agent.name and agent.description for agent in managed_agents), (
                "All managed agents need both a name and a description!"
            )
            self.managed_agents = {agent.name: agent for agent in managed_agents}
            # Set reference to main agent for managed agents to access original_problem
            for agent in managed_agents:
                agent._main_agent = self

    def _setup_tools(self, tools, add_base_tools):
        assert all(isinstance(tool, Tool) for tool in tools), "All elements must be instance of Tool (or a subclass)"
        self.tools = {tool.name: tool for tool in tools}
        if add_base_tools:
            self.tools.update(
                {
                    name: cls()
                    for name, cls in TOOL_MAPPING.items()
                    if name != "python_interpreter" or self.__class__.__name__ == "ToolCallingAgent"
                }
            )
        self.tools.setdefault("final_answer", FinalAnswerTool())

    def _validate_tools_and_managed_agents(self, tools, managed_agents):
        tool_and_managed_agent_names = [tool.name for tool in tools]
        if managed_agents is not None:
            tool_and_managed_agent_names += [agent.name for agent in managed_agents]
        if self.name:
            tool_and_managed_agent_names.append(self.name)
        if len(tool_and_managed_agent_names) != len(set(tool_and_managed_agent_names)):
            raise ValueError(
                "Each tool or managed_agent should have a unique name! You passed these duplicate names: "
                f"{[name for name in tool_and_managed_agent_names if tool_and_managed_agent_names.count(name) > 1]}"
            )

    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: Optional[List[str]] = None,
        additional_args: Optional[Dict] = None,
        max_steps: Optional[int] = None,
    ):
        """
        Run the agent for the given task.

        Args:
            task (`str`): Task to perform.
            stream (`bool`): Whether to run in a streaming way.
            reset (`bool`): Whether to reset the conversation or keep it going from previous run.
            images (`list[str]`, *optional*): Paths to image(s).
            additional_args (`dict`, *optional*): Any other variables that you want to pass to the agent run, for instance images or dataframes. Give them clear names!
            max_steps (`int`, *optional*): Maximum number of steps the agent can take to solve the task. if not provided, will use the agent's default value.

        Example:
        ```py
        from smolagents import CodeAgent
        agent = CodeAgent(tools=[])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """
        max_steps = max_steps or self.max_steps
        self.task = task
        
        # Set original problem only when reset=True and original_problem is not already set
        if reset and self.original_problem is None:
            self.original_problem = task
            # Also propagate to all managed agents
            for agent in self.managed_agents.values():
                if agent.original_problem is None:
                    agent.original_problem = task
        
        if additional_args is not None:
            self.state.update(additional_args)
            self.task += f"""
You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
{str(additional_args)}."""

        self.system_prompt = self.initialize_system_prompt()
        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        if reset:
            self.memory.reset()
            self.monitor.reset()

        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )
        self.memory.steps.append(TaskStep(task=self.task, task_images=images))

        if getattr(self, "python_executor", None):
            self.python_executor.send_variables(variables=self.state)
            self.python_executor.send_tools({**self.tools, **self.managed_agents})

        if stream:
            # The steps are returned as they are executed through a generator to iterate on.
            return self._run(task=self.task, max_steps=max_steps, images=images)
        # Outputs are returned only at the end. We only look at the last step.
        return deque(self._run(task=self.task, max_steps=max_steps, images=images), maxlen=1)[0]

    def _run(
        self, task: str, max_steps: int, images: List[str] | None = None
    ) -> Generator[ActionStep | AgentType, None, None]:
        final_answer = None
        self.step_number = 1
        # Update self.max_steps to ensure dynamic adjustment works
        self.max_steps = max_steps
        while final_answer is None and self.step_number <= self.max_steps:
            step_start_time = time.time()
            memory_step = self._create_memory_step(step_start_time, images)
            try:
                final_answer = self._execute_step(task, memory_step)
            except AgentError as e:
                memory_step.error = e
            finally:
                self._finalize_step(memory_step, step_start_time)
                yield memory_step
                self.step_number += 1

        if final_answer is None and self.step_number == self.max_steps + 1:
            final_answer = self._handle_max_steps_reached(task, images, step_start_time)
            yield memory_step
        yield handle_agent_output_types(final_answer)

    def _create_memory_step(self, step_start_time: float, images: List[str] | None) -> ActionStep:
        return ActionStep(step_number=self.step_number, start_time=step_start_time, observations_images=images)

    def _execute_step(self, task: str, memory_step: ActionStep) -> Union[None, Any]:
        # Check planning interval and step number for planning
        if self.planning_interval is not None and self.step_number % self.planning_interval == 1:
            # Only call planning_step if it's NOT the first step OR if initial task decomposition is enabled
            if not (self.step_number == 1 and not self.enable_initial_task_decomposition):
                self.planning_step(task, is_first_step=(self.step_number == 1), step=self.step_number)
            # If this is the first step and we just did task decomposition, 
            # skip the regular step execution and return None to continue to step 2
            if self.step_number == 1:
                return None
        self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
        
        # Check if this is a validation step
        if getattr(self, '_should_run_validation', False):
            self._should_run_validation = False
            return self._run_validation_step(memory_step)
        
        # Check if this is a format checking step
        if getattr(self, '_should_run_format_check', False):
            self._should_run_format_check = False
            return self._run_format_check_step(memory_step)
        
        final_answer = self.step(memory_step)
        
        # If we got a final answer and should run validation, schedule it for next step
        if final_answer is not None and not self.first_final_answer_called and self.validation_agent and self.original_problem:
            self.first_final_answer_called = True
            self._should_run_validation = True
            self._final_answer_for_validation = final_answer
            # Don't run format checking yet, just return None to continue to validation step
            return None
        
        # If validation has been completed and this is the second final_answer call, trigger format checking
        if (final_answer is not None and self.first_final_answer_called and 
            getattr(self, '_validation_completed', False) and 
            isinstance(self, CodeAgent) and hasattr(self, 'rethink_model') and self.rethink_model):
            # Store current memory step for rethink context
            self._current_memory_step = memory_step
            try:
                # Use CodeAgent's format checking rethink
                format_result = self._rethink_format_checking_code(final_answer)
                self.logger.log(
                    Text(f"Format checking completed. Final answer: {format_result}", style="bold green"),
                    level=LogLevel.INFO
                )
                return format_result
            finally:
                # Clean up current memory step reference
                self._current_memory_step = None
        
        # For CodeAgent with rethink_model but no validation_agent, use the rethink system for format checking only
        if (final_answer is not None and not self.first_final_answer_called and 
            isinstance(self, CodeAgent) and hasattr(self, 'rethink_model') and self.rethink_model and 
            not self.validation_agent):
            self.first_final_answer_called = True
            # Store current memory step for rethink context
            self._current_memory_step = memory_step
            try:
                # Only use format checking part of rethink, skip reasoning validation
                rethink_result = self._rethink_format_checking_code(final_answer)
                return rethink_result
            finally:
                # Clean up current memory step reference
                self._current_memory_step = None
        
        # Only run format checking if validation has been completed (or if validation is not enabled)
        if final_answer is not None and self.final_answer_checks:
            # If validation is enabled but hasn't been completed yet, skip format checking
            if self.validation_agent and self.original_problem and not getattr(self, '_validation_completed', False):
                # Skip format checking until after validation
                pass
            else:
                # Either validation is disabled or validation has been completed, run format checking
                self._validate_final_answer(final_answer)
        return final_answer
    


    def _validate_final_answer(self, final_answer: Any):
        for check_function in self.final_answer_checks:
            try:
                assert check_function(final_answer, self.memory)
            except Exception as e:
                raise AgentError(f"Check {check_function.__name__} failed with error: {e}", self.logger)

    def _run_validation_step(self, memory_step: ActionStep) -> Union[None, Any]:
        """Run ValidationAgent as a separate step"""
        try:
            # Mark relevant steps for validation
            self._mark_validation_steps()
            
            max_call_num = 0
            for memory_step_item in self.memory.steps:
                if hasattr(memory_step_item, 'tool_calls') and memory_step_item.tool_calls:
                    for tool_call in memory_step_item.tool_calls:
                        if tool_call.id.startswith("call_"):
                            try:
                                call_num = int(tool_call.id.split("_")[1])
                                max_call_num = max(max_call_num, call_num)
                            except (ValueError, IndexError):
                                pass
            
            validation_call_id = f"call_{max_call_num + 1}"
            
            # Create a mock tool call for validation - import from correct location
            from smolagents.memory import ToolCall
            validation_tool_call = ToolCall(
                id=validation_call_id,
                name="validation_agent",
                arguments="Validating reasoning steps"
            )
            
            # Set up the memory step to look like a validation tool call
            memory_step.model_output = """Thought: Call the validation agent to check the answer.\nCode:\n```py\nvalidation_agent("Validating reasoning steps")\n```"""
            memory_step.tool_calls = [validation_tool_call]
            
            validation_feedback = self.validation_agent.validate_answer(self, str(self._final_answer_for_validation))
            
            # Create validation observation (the call_id will be added automatically by to_messages)
            validation_observation = f"\nExecution logs:\nLast output from code snippet:\nHere is feedback from 'validation agent': {validation_feedback}"
            memory_step.observations = validation_observation
            memory_step.action_output = None  # No action output since this is just validation
            
            # If validation found issues, continue conversation
            if validation_feedback != "All reasoning steps are correct.":
                # Mark validation as completed (even with errors) so second final_answer can trigger format checking
                self._validation_completed = True
                
                # Adjust max_steps to allow for more steps
                remaining_steps = max(5, self.max_steps - self.step_number + 5)
                self.max_steps = max(self.max_steps, self.step_number + remaining_steps)
                
                self.logger.log(
                    Text(f"Validation feedback with suggestions: {validation_feedback}", style="bold red"),
                    level=LogLevel.INFO
                )
                
                # Update the observation to include detailed validation suggestions
                detailed_feedback = f"""Validation Agent feedback: The reasoning has some issues that need to be addressed.

{validation_feedback}

Please review and correct the reasoning based on these suggestions, then provide a corrected final answer."""
                
                memory_step.observations = detailed_feedback
                
                # Return None to continue the conversation with the validation feedback
                return None
            else:
                self.logger.log(
                    Text("Validation passed: All reasoning steps are correct.", style="bold green"),
                    level=LogLevel.INFO
                )
                # Mark validation as completed and directly trigger format checking
                self._validation_completed = True
                
                # For CodeAgent with rethink_model, directly use its format checking system
                if isinstance(self, CodeAgent) and hasattr(self, 'rethink_model') and self.rethink_model:
                    # Store current memory step for rethink context
                    self._current_memory_step = memory_step
                    try:
                        # Use CodeAgent's format checking rethink
                        format_result = self._rethink_format_checking_code(self._final_answer_for_validation)
                        self.logger.log(
                            Text(f"Format checking completed. Final answer: {format_result}", style="bold green"),
                            level=LogLevel.INFO
                        )
                        return format_result
                    finally:
                        # Clean up current memory step reference
                        self._current_memory_step = None
                else:
                    # For other agents, run standard format checking
                    if self.final_answer_checks:
                        try:
                            self._validate_final_answer(self._final_answer_for_validation)
                            self.logger.log(
                                Text(f"Format checking completed. Final answer: {self._final_answer_for_validation}", style="bold green"),
                                level=LogLevel.INFO
                            )
                            return self._final_answer_for_validation
                        except Exception as e:
                            self.logger.log(f"Format checking error: {e}", level=LogLevel.ERROR)
                            return self._final_answer_for_validation
                    else:
                        # No format checks defined, return the validated answer
                        return self._final_answer_for_validation
                
        except Exception as e:
            self.logger.log(f"Validation error: {e}", level=LogLevel.ERROR)
            # Mark validation as completed, but let the agent continue running
            # Format checking will be triggered when the agent calls final_answer again
            self._validation_completed = True
            
            # Return None to let the agent continue its execution
            # Format checking will be triggered on the next final_answer call
            return None

    def _run_format_check_step(self, memory_step: ActionStep):
        """
        Run format checking as a separate step after validation.
        """
        try:
            self.logger.log(
                Text("Initiating format checking ...", style="bold blue"),
                level=LogLevel.INFO
            )
            
            # Run format checking on the validated answer
            if self.final_answer_checks:
                self._validate_final_answer(self._final_answer_for_validation)
            
            self.logger.log(
                Text("Format checking completed. Final answer: " + str(self._final_answer_for_validation), style="bold green"),
                level=LogLevel.INFO
            )
            
            # Return the final answer
            return self._final_answer_for_validation
            
        except Exception as e:
            self.logger.log(f"Format checking error: {e}", level=LogLevel.ERROR)
            # Return the original final answer on format checking error
            return self._final_answer_for_validation

    def _mark_validation_steps(self):
        """
        Mark memory steps that should be included in validation.
        Only includes step 0 (original task) and python_interpreter calls (except final_answer).
        """
        for memory_step in self.memory.steps:
            if hasattr(memory_step, 'include_in_validation'):
                # Check if this is a python_interpreter call
                if hasattr(memory_step, 'tool_calls') and memory_step.tool_calls:
                    for tool_call in memory_step.tool_calls:
                        if tool_call.name == "python_interpreter":
                            # Check if this is NOT a final_answer call
                            arguments = tool_call.arguments
                            if isinstance(arguments, str) and "final_answer" not in arguments.lower():
                                memory_step.include_in_validation = True
                                break

    def _finalize_step(self, memory_step: ActionStep, step_start_time: float):
        memory_step.end_time = time.time()
        memory_step.duration = memory_step.end_time - step_start_time
        self.memory.steps.append(memory_step)
        for callback in self.step_callbacks:
            # For compatibility with old callbacks that don't take the agent as an argument
            callback(memory_step) if len(inspect.signature(callback).parameters) == 1 else callback(
                memory_step, agent=self
            )

    def _handle_max_steps_reached(self, task: str, images: List[str], step_start_time: float) -> Any:
        final_answer = self.provide_final_answer(task, images)
        final_memory_step = ActionStep(
            step_number=self.step_number, error=AgentMaxStepsError("Reached max steps.", self.logger)
        )
        final_memory_step.action_output = final_answer
        final_memory_step.end_time = time.time()
        final_memory_step.duration = final_memory_step.end_time - step_start_time
        self.memory.steps.append(final_memory_step)
        for callback in self.step_callbacks:
            callback(final_memory_step) if len(inspect.signature(callback).parameters) == 1 else callback(
                final_memory_step, agent=self
            )
        return final_answer

    def planning_step(self, task, is_first_step: bool, step: int) -> None:
        if is_first_step:
            input_messages, decomposition_message = self._generate_initial_decomposition(task, step)
            self._record_planning_step(input_messages, None, None, is_first_step, decomposition_message)
        else:
            input_messages, facts_message, plan_message = self._generate_updated_plan(task, step)
            self._record_planning_step(input_messages, facts_message, plan_message, is_first_step, None)

    def _generate_initial_decomposition(self, task: str, step: int) -> Tuple[List[Dict], ChatMessage]:
        input_messages = [
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["task_decomposition"], 
                            variables={
                                "task": task,
                                "tools": self.tools,
                                "managed_agents": self.managed_agents,
                                "remaining_steps": (self.max_steps - step),
                            }
                        ),
                    }
                ],
            },
        ]
        decomposition_message = self.model(input_messages)
        return input_messages, decomposition_message

    def _generate_updated_plan(self, task: str, step: int) -> Tuple[ChatMessage, ChatMessage]:
        # Do not take the system prompt message from the memory
        # summary_mode=False: Do not take previous plan steps to avoid influencing the new plan
        memory_messages = self.write_memory_to_messages()[1:]
        facts_update_pre = {
            "role": MessageRole.SYSTEM,
            "content": [{"type": "text", "text": self.prompt_templates["planning"]["update_facts_pre_messages"]}],
        }
        facts_update_post = {
            "role": MessageRole.USER,
            "content": [{"type": "text", "text": self.prompt_templates["planning"]["update_facts_post_messages"]}],
        }
        input_messages = [facts_update_pre] + memory_messages + [facts_update_post]
        facts_message = self.model(input_messages)

        update_plan_pre = {
            "role": MessageRole.SYSTEM,
            "content": [
                {
                    "type": "text",
                    "text": populate_template(
                        self.prompt_templates["planning"]["update_plan_pre_messages"], variables={"task": task}
                    ),
                }
            ],
        }
        update_plan_post = {
            "role": MessageRole.USER,
            "content": [
                {
                    "type": "text",
                    "text": populate_template(
                        self.prompt_templates["planning"]["update_plan_post_messages"],
                        variables={
                            "task": task,
                            "tools": self.tools,
                            "managed_agents": self.managed_agents,
                            "facts_update": facts_message.content,
                            "remaining_steps": (self.max_steps - step),
                        },
                    ),
                }
            ],
        }
        plan_message = self.model(
            [update_plan_pre] + memory_messages + [update_plan_post], stop_sequences=["<end_plan>"]
        )
        return input_messages, facts_message, plan_message

    def _record_planning_step(
        self, input_messages: list, facts_message: ChatMessage, plan_message: ChatMessage, is_first_step: bool, decomposition_message: ChatMessage = None
    ) -> None:
        if is_first_step:
            decomposed_tasks = textwrap.dedent(f"""Here is the task decomposition:\n```\n{decomposition_message.content}\n```""")
            log_message = "Task decomposition"
            display_content = decomposed_tasks
            
            self.memory.steps.append(
                PlanningStep(
                    model_input_messages=input_messages,
                    decomposed_tasks=decomposed_tasks,
                    model_output_message_decomposition=decomposition_message,
                )
            )
        else:
            facts = textwrap.dedent(
                f"""Here is the updated list of the facts that I know:\n```\n{facts_message.content}\n```"""
            )
            plan = textwrap.dedent(
                f"""I still need to solve the task:\n```\n{self.task}\n```\n\nHere is the new/updated plan of action to solve the task:\n```\n{plan_message.content}\n```"""
            )
            log_message = "Updated plan"
            display_content = plan
            
            self.memory.steps.append(
                PlanningStep(
                    model_input_messages=input_messages,
                    facts=facts,
                    plan=plan,
                    model_output_message_plan=plan_message,
                    model_output_message_facts=facts_message,
                )
            )
        
        self.logger.log(Rule(f"[bold]{log_message}", style="orange"), Text(display_content), level=LogLevel.INFO)

    #获取最新的plan
    def get_latest_plan(self) -> str:
        """
        Retrieves the most recent plan from a PlanningStep in memory.
        Only PlanningSteps has a plan attribute.
        If the most recent step is an ActionStep (which is common during the agent's execution phase), it doesn't have a plan attribute, leading to an AttributeError.
        """
        try:
            # First look for the most recent updated plan
            for step in reversed(self.memory.steps):
                if isinstance(step, PlanningStep) and step.plan:
                    return step.plan
            
            # If no updated plan found, look for the initial decomposed tasks
            for step in reversed(self.memory.steps):
                if isinstance(step, PlanningStep) and step.decomposed_tasks:
                    return step.decomposed_tasks
                    
            return ""
        except Exception as e:
            raise AgentError(f"Error in getting latest plan: {e}", self.logger)

    @property
    def logs(self):
        logger.warning(
            "The 'logs' attribute is deprecated and will soon be removed. Please use 'self.memory.steps' instead."
        )
        return [self.memory.system_prompt] + self.memory.steps

    def initialize_system_prompt(self):
        """To be implemented in child classes"""
        pass

    def write_memory_to_messages(
        self,
        summary_mode: Optional[bool] = False,
    ) -> List[Dict[str, str]]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        return messages

    def visualize(self):
        """Creates a rich tree visualization of the agent's structure."""
        self.logger.visualize_agent_tree(self)

    def extract_action(self, model_output: str, split_token: str) -> Tuple[str, str]:
        """
        Parse action from the LLM output

        Args:
            model_output (`str`): Output of the LLM
            split_token (`str`): Separator for the action. Should match the example in the system prompt.
        """
        try:
            split = model_output.split(split_token)
            rationale, action = (
                split[-2],
                split[-1],
            )  # NOTE: using indexes starting from the end solves for when you have more than one split_token in the output
        except Exception:
            raise AgentParsingError(
                f"No '{split_token}' token provided in your output.\nYour output:\n{model_output}\n. Be sure to include an action, prefaced with '{split_token}'!",
                self.logger,
            )
        return rationale.strip(), action.strip()

    def provide_final_answer(self, task: str, images: Optional[list[str]]) -> str:
        """
        Provide the final answer to the task, based on the logs of the agent's interactions.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`, *optional*): Paths to image(s).

        Returns:
            `str`: Final answer to the task.
        """
        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_templates["final_answer"]["pre_messages"],
                    }
                ],
            }
        ]
        if images:
            messages[0]["content"].append({"type": "image"})
        messages += self.write_memory_to_messages()[1:]
        messages += [
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["final_answer"]["post_messages"], variables={"task": task}
                        ),
                    }
                ],
            }
        ]
        try:
            chat_message: ChatMessage = self.model(messages)
            return chat_message.content
        except Exception as e:
            return f"Error in generating final LLM output:\n{e}"

    def execute_tool_call(self, tool_name: str, arguments: Union[Dict[str, str], str]) -> Any:
        """
        Execute tool with the provided input and returns the result.
        This method replaces arguments with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the Tool to execute (should be one from self.tools).
            arguments (Dict[str, str]): Arguments passed to the Tool.
        """
        available_tools = {**self.tools, **self.managed_agents}
        if tool_name not in available_tools:
            error_msg = f"Unknown tool {tool_name}, should be instead one of {list(available_tools.keys())}."
            raise AgentExecutionError(error_msg, self.logger)

        try:
            if isinstance(arguments, str):
                if tool_name in self.managed_agents:
                    observation = available_tools[tool_name].__call__(arguments, _calling_agent=self)
                else:
                    observation = available_tools[tool_name].__call__(arguments, sanitize_inputs_outputs=True)
            elif isinstance(arguments, dict):
                for key, value in arguments.items():
                    if isinstance(value, str) and value in self.state:
                        arguments[key] = self.state[value]
                if tool_name in self.managed_agents:
                    observation = available_tools[tool_name].__call__(_calling_agent=self, **arguments)
                else:
                    observation = available_tools[tool_name].__call__(**arguments, sanitize_inputs_outputs=True)
            else:
                error_msg = f"Arguments passed to tool should be a dict or string: got a {type(arguments)}."
                raise AgentExecutionError(error_msg, self.logger)
            return observation
        except Exception as e:
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                error_msg = (
                    f"Error when executing tool {tool_name} with arguments {arguments}: {type(e).__name__}: {e}\nYou should only use this tool with a correct input.\n"
                    f"As a reminder, this tool's description is the following: '{tool.description}'.\nIt takes inputs: {tool.inputs} and returns output type {tool.output_type}"
                )
                raise AgentExecutionError(error_msg, self.logger)
            elif tool_name in self.managed_agents:
                error_msg = (
                    f"Error in calling team member: {e}\nYou should only ask this team member with a correct request.\n"
                    f"As a reminder, this team member's description is the following:\n{available_tools[tool_name]}"
                )
                raise AgentExecutionError(error_msg, self.logger)

    def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """To be implemented in children classes. Should return either None if the step is not final."""
        pass

    def replay(self, detailed: bool = False):
        """Prints a pretty replay of the agent's steps.

        Args:
            detailed (bool, optional): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        self.memory.replay(self.logger, detailed=detailed)

    def __call__(self, task: str, **kwargs):
        """Adds additional prompting for the managed agent, runs it, and wraps the output.
        This method is called only by a managed agent.
        """
        # Get original problem from self or from main agent
        original_problem = getattr(self, 'original_problem', None) or ""
        if not original_problem and hasattr(self, '_main_agent') and self._main_agent:
            original_problem = getattr(self._main_agent, 'original_problem', '') or ""
        
        full_task = populate_template(
            self.prompt_templates["managed_agent"]["task"],
            variables=dict(name=self.name, task=task, problem=original_problem),
        )
        
        # Preserve the current original_problem to avoid it being overwritten by reset=True
        preserved_original_problem = self.original_problem
        
        report = self.run(full_task, reset=True, **kwargs)
        
        # Restore the original_problem after reset
        self.original_problem = preserved_original_problem
        
        answer = populate_template(
            self.prompt_templates["managed_agent"]["report"], variables=dict(name=self.name, final_answer=report)
        )
        return answer

    def save(self, output_dir: str, relative_path: Optional[str] = None):
        """
        Saves the relevant code files for your agent. This will copy the code of your agent in `output_dir` as well as autogenerate:

        - a `tools` folder containing the logic for each of the tools under `tools/{tool_name}.py`.
        - a `managed_agents` folder containing the logic for each of the managed agents.
        - an `agent.json` file containing a dictionary representing your agent.
        - a `prompt.yaml` file containing the prompt templates used by your agent.
        - an `app.py` file providing a UI for your agent when it is exported to a Space with `agent.push_to_hub()`
        - a `requirements.txt` containing the names of the modules used by your tool (as detected when inspecting its
          code)

        Args:
            output_dir (`str`): The folder in which you want to save your tool.
        """
        make_init_file(output_dir)

        # Recursively save managed agents
        if self.managed_agents:
            make_init_file(os.path.join(output_dir, "managed_agents"))
            for agent_name, agent in self.managed_agents.items():
                agent_suffix = f"managed_agents.{agent_name}"
                if relative_path:
                    agent_suffix = relative_path + "." + agent_suffix
                agent.save(os.path.join(output_dir, "managed_agents", agent_name), relative_path=agent_suffix)

        class_name = self.__class__.__name__

        # Save tools to different .py files
        for tool in self.tools.values():
            make_init_file(os.path.join(output_dir, "tools"))
            tool.save(os.path.join(output_dir, "tools"), tool_file_name=tool.name, make_gradio_app=False)

        # Save prompts to yaml
        yaml_prompts = yaml.safe_dump(
            self.prompt_templates,
            default_style="|",  # This forces block literals for all strings
            default_flow_style=False,
            width=float("inf"),
            sort_keys=False,
            allow_unicode=True,
            indent=2,
        )

        with open(os.path.join(output_dir, "prompts.yaml"), "w", encoding="utf-8") as f:
            f.write(yaml_prompts)

        # Save agent dictionary to json
        agent_dict = self.to_dict()
        agent_dict["tools"] = [tool.name for tool in self.tools.values()]
        with open(os.path.join(output_dir, "agent.json"), "w", encoding="utf-8") as f:
            json.dump(agent_dict, f, indent=4)

        # Save requirements
        with open(os.path.join(output_dir, "requirements.txt"), "w", encoding="utf-8") as f:
            f.writelines(f"{r}\n" for r in agent_dict["requirements"])

        # Make agent.py file with Gradio UI
        agent_name = f"agent_{self.name}" if getattr(self, "name", None) else "agent"
        managed_agent_relative_path = relative_path + "." if relative_path is not None else ""
        app_template = textwrap.dedent("""
            import yaml
            import os
            from smolagents import GradioUI, {{ class_name }}, {{ agent_dict['model']['class'] }}

            # Get current directory path
            CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

            {% for tool in tools.values() -%}
            from {{managed_agent_relative_path}}tools.{{ tool.name }} import {{ tool.__class__.__name__ }} as {{ tool.name | camelcase }}
            {% endfor %}
            {% for managed_agent in managed_agents.values() -%}
            from {{managed_agent_relative_path}}managed_agents.{{ managed_agent.name }}.app import agent_{{ managed_agent.name }}
            {% endfor %}

            model = {{ agent_dict['model']['class'] }}(
            {% for key in agent_dict['model']['data'] if key not in ['class', 'last_input_token_count', 'last_output_token_count'] -%}
                {{ key }}={{ agent_dict['model']['data'][key]|repr }},
            {% endfor %})

            {% for tool in tools.values() -%}
            {{ tool.name }} = {{ tool.name | camelcase }}()
            {% endfor %}

            with open(os.path.join(CURRENT_DIR, "prompts.yaml"), 'r') as stream:
                prompt_templates = yaml.safe_load(stream)

            {{ agent_name }} = {{ class_name }}(
                model=model,
                tools=[{% for tool_name in tools.keys() if tool_name != "final_answer" %}{{ tool_name }}{% if not loop.last %}, {% endif %}{% endfor %}],
                managed_agents=[{% for subagent_name in managed_agents.keys() %}agent_{{ subagent_name }}{% if not loop.last %}, {% endif %}{% endfor %}],
                {% for attribute_name, value in agent_dict.items() if attribute_name not in ["model", "tools", "prompt_templates", "authorized_imports", "managed_agents", "requirements"] -%}
                {{ attribute_name }}={{ value|repr }},
                {% endfor %}prompt_templates=prompt_templates
            )
            if __name__ == "__main__":
                GradioUI({{ agent_name }}).launch()
            """).strip()
        template_env = jinja2.Environment(loader=jinja2.BaseLoader(), undefined=jinja2.StrictUndefined)
        template_env.filters["repr"] = repr
        template_env.filters["camelcase"] = lambda value: "".join(word.capitalize() for word in value.split("_"))
        template = template_env.from_string(app_template)

        # Render the app.py file from Jinja2 template
        app_text = template.render(
            {
                "agent_name": agent_name,
                "class_name": class_name,
                "agent_dict": agent_dict,
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "managed_agent_relative_path": managed_agent_relative_path,
            }
        )

        with open(os.path.join(output_dir, "app.py"), "w", encoding="utf-8") as f:
            f.write(app_text + "\n")  # Append newline at the end

    def to_dict(self) -> Dict[str, Any]:
        """Converts agent into a dictionary."""
        # TODO: handle serializing step_callbacks and final_answer_checks
        for attr in ["final_answer_checks", "step_callbacks"]:
            if getattr(self, attr, None):
                self.logger.log(f"This agent has {attr}: they will be ignored by this method.", LogLevel.INFO)

        tool_dicts = [tool.to_dict() for tool in self.tools.values()]
        tool_requirements = {req for tool in self.tools.values() for req in tool.to_dict()["requirements"]}
        managed_agents_requirements = {
            req for managed_agent in self.managed_agents.values() for req in managed_agent.to_dict()["requirements"]
        }
        requirements = tool_requirements | managed_agents_requirements
        if hasattr(self, "authorized_imports"):
            requirements.update(
                {package.split(".")[0] for package in self.authorized_imports if package not in BASE_BUILTIN_MODULES}
            )

        agent_dict = {
            "tools": tool_dicts,
            "model": {
                "class": self.model.__class__.__name__,
                "data": self.model.to_dict(),
            },
            "managed_agents": {
                managed_agent.name: managed_agent.__class__.__name__ for managed_agent in self.managed_agents.values()
            },
            "prompt_templates": self.prompt_templates,
            "max_steps": self.max_steps,
            "verbosity_level": int(self.logger.level),
            "grammar": self.grammar,
            "planning_interval": self.planning_interval,
            "name": self.name,
            "description": self.description,
            "requirements": list(requirements),
        }
        if hasattr(self, "authorized_imports"):
            agent_dict["authorized_imports"] = self.authorized_imports
        if hasattr(self, "executor_type"):
            agent_dict["executor_type"] = self.executor_type
            agent_dict["executor_kwargs"] = self.executor_kwargs
        if hasattr(self, "max_print_outputs_length"):
            agent_dict["max_print_outputs_length"] = self.max_print_outputs_length
        return agent_dict

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        token: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Loads an agent defined on the Hub.

        <Tip warning={true}>

        Loading a tool from the Hub means that you'll download the tool and execute it locally.
        ALWAYS inspect the tool you're downloading before loading it within your runtime, as you would do when
        installing a package using pip/npm/apt.

        </Tip>

        Args:
            repo_id (`str`):
                The name of the repo on the Hub where your tool is defined.
            token (`str`, *optional*):
                The token to identify you on hf.co. If unset, will use the token generated when running
                `huggingface-cli login` (stored in `~/.huggingface`).
            trust_remote_code(`bool`, *optional*, defaults to False):
                This flags marks that you understand the risk of running remote code and that you trust this tool.
                If not setting this to True, loading the tool from Hub will fail.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
                `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your agent, and the
                others will be passed along to its init.
        """
        if not trust_remote_code:
            raise ValueError(
                "Loading an agent from Hub requires to acknowledge you trust its code: to do so, pass `trust_remote_code=True`."
            )

        # Get the agent's Hub folder.
        download_kwargs = {"token": token, "repo_type": "space"} | {
            key: kwargs.pop(key)
            for key in [
                "cache_dir",
                "force_download",
                "proxies",
                "revision",
                "local_files_only",
            ]
            if key in kwargs
        }

        download_folder = Path(snapshot_download(repo_id=repo_id, **download_kwargs))
        return cls.from_folder(download_folder, **kwargs)

    @classmethod
    def from_folder(cls, folder: Union[str, Path], **kwargs):
        """Loads an agent from a local folder.

        Args:
            folder (`str` or `Path`): The folder where the agent is saved.
            **kwargs: Additional keyword arguments that will be passed to the agent's init.
        """
        folder = Path(folder)
        agent_dict = json.loads((folder / "agent.json").read_text())

        # Recursively get managed agents
        managed_agents = []
        for managed_agent_name, managed_agent_class in agent_dict["managed_agents"].items():
            agent_cls = getattr(importlib.import_module("smolagents.agents"), managed_agent_class)
            managed_agents.append(agent_cls.from_folder(folder / "managed_agents" / managed_agent_name))

        tools = []
        for tool_name in agent_dict["tools"]:
            tool_code = (folder / "tools" / f"{tool_name}.py").read_text()
            tools.append(Tool.from_code(tool_code))

        model_class: Model = getattr(importlib.import_module("smolagents.models"), agent_dict["model"]["class"])
        model = model_class.from_dict(agent_dict["model"]["data"])

        args = dict(
            model=model,
            tools=tools,
            managed_agents=managed_agents,
            name=agent_dict["name"],
            description=agent_dict["description"],
            max_steps=agent_dict["max_steps"],
            planning_interval=agent_dict["planning_interval"],
            grammar=agent_dict["grammar"],
            verbosity_level=agent_dict["verbosity_level"],
        )
        if cls.__name__ == "CodeAgent":
            args["additional_authorized_imports"] = agent_dict["authorized_imports"]
            args["executor_type"] = agent_dict["executor_type"]
            args["executor_kwargs"] = agent_dict["executor_kwargs"]
            args["max_print_outputs_length"] = agent_dict["max_print_outputs_length"]
        args.update(kwargs)
        return cls(**args)

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload agent",
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
    ) -> str:
        """
        Upload the agent to the Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push to. It should contain your organization name when
                pushing to a given organization.
            commit_message (`str`, *optional*, defaults to `"Upload agent"`):
                Message to commit while pushing.
            private (`bool`, *optional*, defaults to `None`):
                Whether to make the repo private. If `None`, the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether to create a PR with the uploaded files or directly commit.
        """
        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="space",
            space_sdk="gradio",
        )
        repo_id = repo_url.repo_id
        metadata_update(
            repo_id,
            {"tags": ["smolagents", "agent"]},
            repo_type="space",
            token=token,
            overwrite=True,
        )

        with tempfile.TemporaryDirectory() as work_dir:
            self.save(work_dir)
            logger.info(f"Uploading the following files to {repo_id}: {','.join(os.listdir(work_dir))}")
            return upload_folder(
                repo_id=repo_id,
                commit_message=commit_message,
                folder_path=work_dir,
                token=token,
                create_pr=create_pr,
                repo_type="space",
            )

    def set_original_problem(self, original_problem: str):
        """Set the original problem for this agent and all its managed agents.
        This should be called before running the agent if you want managed agents 
        to have access to the original problem context.
        
        Args:
            original_problem (str): The original problem statement
        """
        self.original_problem = original_problem
        
        # Also set for managed agents
        if hasattr(self, 'managed_agents') and self.managed_agents:
            for agent in self.managed_agents:
                if hasattr(agent, 'set_original_problem'):
                    agent.set_original_problem(original_problem)

    def _mark_validation_steps(self):
        """
        Mark memory steps that should be included in validation.
        Only includes step 0 (original task) and python_interpreter calls (except final_answer).
        """
        for memory_step in self.memory.steps:
            if hasattr(memory_step, 'include_in_validation'):
                # Check if this is a python_interpreter call
                if hasattr(memory_step, 'tool_calls') and memory_step.tool_calls:
                    for tool_call in memory_step.tool_calls:
                        if tool_call.name == "python_interpreter":
                            # Check if this is NOT a final_answer call
                            arguments = tool_call.arguments
                            if isinstance(arguments, str) and "final_answer" not in arguments.lower():
                                memory_step.include_in_validation = True
                                break


class ToolCallingAgent(MultiStepAgent):
    """
    This agent uses JSON-like tool calls, using method `model.get_tool_call` to leverage the LLM engine's tool calling capabilities.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        rethink_model (`Callable[[list[dict[str, str]]], ChatMessage]`, *optional*): Model for reflection and verification.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        tools: List[Tool],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        prompt_templates: Optional[PromptTemplates] = None,
        planning_interval: Optional[int] = None,
        rethink_model: Optional[Callable[[List[Dict[str, str]]], ChatMessage]] = None,
        **kwargs,
    ):
        prompt_templates = prompt_templates or yaml.safe_load(
            importlib.resources.files("smolagents.prompts").joinpath("toolcalling_agent.yaml").read_text()
        )
        self.rethink_model = rethink_model
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            **kwargs,
        )

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={"tools": self.tools, "managed_agents": self.managed_agents},
        )
        return system_prompt

    def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        memory_messages = self.write_memory_to_messages()

        self.input_messages = memory_messages

        # Add new step in logs
        memory_step.model_input_messages = memory_messages.copy()

        try:
            model_message: ChatMessage = self.model(
                memory_messages,
                tools_to_call_from=list(self.tools.values()),
                stop_sequences=["Observation:"],
            )
            memory_step.model_output_message = model_message
            if model_message.tool_calls is None or len(model_message.tool_calls) == 0:
                raise Exception("Model did not call any tools. Call `final_answer` tool to return a final answer.")

            memory_step.tool_calls = []
            # 遍历所有的工具调用信息
            for tool_call in model_message.tool_calls:
                tool_name, tool_call_id = tool_call.function.name, tool_call.id
                tool_arguments = tool_call.function.arguments

                memory_step.tool_calls.append(ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call_id))

                self.logger.log(
                    Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
                    level=LogLevel.INFO,
                )
                if tool_name == "final_answer":
                    if isinstance(tool_arguments, dict):
                        if "answer" in tool_arguments:
                            answer = tool_arguments["answer"]
                        else:
                            answer = tool_arguments
                    else:
                        answer = tool_arguments
                    if isinstance(answer, str) and answer in self.state.keys():
                        final_answer = self.state[answer]
                        self.logger.log(
                            f"[bold {YELLOW_HEX}]Final answer:[/bold {YELLOW_HEX}] Extracting key '{answer}' from state to return value '{final_answer}'.",
                            level=LogLevel.INFO,
                        )
                    else:
                        final_answer = answer
                        self.logger.log(
                            Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                            level=LogLevel.INFO,
                        )

                    memory_step.action_output = final_answer
                    
                    if self.rethink_model and not getattr(self, '_in_rethink_process', False):
                        # Store current memory step for rethink context
                        self._current_memory_step = memory_step
                        try:
                            rethink_result = self._rethink_final_answer(final_answer)
                            return rethink_result
                        finally:
                            # Clean up current memory step reference
                            self._current_memory_step = None
                    
                    return final_answer
                else:
                    if tool_arguments is None:
                        tool_arguments = {}
                    observation = self.execute_tool_call(tool_name, tool_arguments)
                    observation_type = type(observation)
                    if observation_type in [AgentImage, AgentAudio]:
                        if observation_type == AgentImage:
                            observation_name = "image.png"
                        elif observation_type == AgentAudio:
                            observation_name = "audio.mp3"
                        # TODO: observation naming could allow for different names of same type
                        self.state[observation_name] = observation
                        updated_information = f"Stored '{observation_name}' in memory."
                    else:
                        updated_information = str(observation).strip()
                    self.logger.log(
                        f"Observations: {updated_information.replace('[', '|')}",  # escape potential rich-tag-like components
                        level=LogLevel.INFO,
                    )
                    if tool_name == "web_search":
                        message_websearch_filter = {
                            "role": MessageRole.USER,
                            "content": [
                                {
                                    "type": "text",
                                    "text": populate_template(
                                        self.prompt_templates["websearch_filter"],
                                        variables={
                                            "task": getattr(self, 'raw_task', self.task),
                                            "latest_plan": self.get_latest_plan(),
                                            "raw_websites": updated_information,
                                        },
                                    ),
                                }
                            ],
                        }
                        filtered_message = self.model([message_websearch_filter])
                        if re.match(r"(?i)^\s*[-\*\s'\"`]*no[-\*\s'\"`]*\s*$", filtered_message.content):   #忽略输出格式的影响，如No的大小写或带有其他符号
                            updated_information = "In this round of searching,the web search tool was unable to find a result that was particularly helpful for the task, please change the arguments and try again."
                        else:
                            updated_information = filtered_message.content   
                        
                        self.logger.log(
                            f"processed search result: {updated_information.replace('[', '|')}",  # escape potential rich-tag-like components
                            level=LogLevel.INFO,
                        )
                    if memory_step.observations is None:
                        memory_step.observations = updated_information  # 初始化为空字符串
                    else:
                        memory_step.observations += "\ntool_call_id:" + tool_call_id + "\nObservation:\n\n" + updated_information

            return None

        except Exception as e:
            raise AgentGenerationError(f"Error in generating tool call with model:\n{e}", self.logger) from e

    def _rethink_final_answer(self, proposed_answer: Any) -> Any:
        """
        Use the rethink model to reflect on the proposed final answer and correct it if needed.
        For ToolCallingAgent, only reasoning validation is performed.
        """
        if not self.rethink_model:
            return proposed_answer
            
        # Set flag to prevent recursive rethink calls
        self._in_rethink_process = True
        
        try:
            # Only Stage 1: Reasoning validation (no format checking for ToolCallingAgent)
            final_answer = self._rethink_reasoning_validation(proposed_answer)
            
            return final_answer
            
        except Exception as e:
            self.logger.log_markdown(
                content=f"❌ Error in rethink process: {e}. Using original answer.",
                title="Rethink Error",
                level=LogLevel.INFO,
            )
            return proposed_answer
            
        finally:
            # Always reset the flag when exiting rethink process
            self._in_rethink_process = False

    def _rethink_reasoning_validation(self, proposed_answer: Any) -> Any:
        """
        Stage: Use the rethink model to validate the reasoning process with XML format.
        """
        try:
            # Build the reasoning context from memory
            reasoning_context = self._build_reasoning_context()
            
            # Create the reasoning rethink prompt
            # For ToolCallingAgent, rethink is a single string, not a dict with "reasoning" key
            rethink_template = self.prompt_templates["rethink"]
            if isinstance(rethink_template, dict) and "reasoning" in rethink_template:
                # CodeAgent style: use the reasoning key
                rethink_prompt = populate_template(
                    rethink_template["reasoning"],
                    variables={
                        "task": getattr(self, 'raw_task', self.task),
                        "reasoning_context": reasoning_context,
                        "proposed_answer": str(proposed_answer),
                    },
                )
            else:
                # ToolCallingAgent style: rethink is a single string
                rethink_prompt = populate_template(
                    rethink_template,
                    variables={
                        "task": getattr(self, 'raw_task', self.task),
                        "reasoning_context": reasoning_context,
                        "proposed_answer": str(proposed_answer),
                    },
                )
            
            # Prepare the input messages for the rethink model
            rethink_messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": rethink_prompt}],
                }
            ]
            
            self.logger.log_markdown(
                content="🤔 Initiating reasoning validation ...",
                title="Tool Calling Rethink",
                level=LogLevel.INFO,
            )
            
            # Call the rethink model without tool calling
            rethink_response = self.rethink_model(
                rethink_messages,
                stop_sequences=["</final_answer>"],
            )
            
            # Record rethink information in current memory step
            if hasattr(self, '_current_memory_step') and self._current_memory_step:
                if not hasattr(self._current_memory_step, 'rethink_info'):
                    self._current_memory_step.rethink_info = {}
                self._current_memory_step.rethink_info.update({
                    'reasoning_rethink_prompt': rethink_prompt,
                    'reasoning_rethink_input_messages': rethink_messages,
                    'reasoning_rethink_response': rethink_response,
                    'reasoning_rethink_content': rethink_response.content,
                    'original_answer': proposed_answer
                })
            
            self.logger.log_markdown(
                content=rethink_response.content,
                title="Reasoning Validation Response",
                level=LogLevel.INFO,
            )
            
            # Parse the XML response to extract final_answer
            try:
                # Extract the final_answer using XML parsing
                reasoning_answer = self._extract_final_answer_xml(rethink_response.content)
                
                if reasoning_answer is not None:
                    # Record reasoning validation results
                    if hasattr(self, '_current_memory_step') and self._current_memory_step and hasattr(self._current_memory_step, 'rethink_info'):
                        self._current_memory_step.rethink_info.update({
                            'reasoning_rethink_final_answer': reasoning_answer,
                            'reasoning_rethink_success': True,
                            'reasoning_rethink_method': 'xml'
                        })
                    
                    self.logger.log_markdown(
                        content=f"✅ Reasoning validation completed (XML).",
                        title="Reasoning Validation Result",
                        level=LogLevel.INFO,
                    )
                    return reasoning_answer
                else:
                    self.logger.log_markdown(
                        content="⚠️ Could not find final_answer in XML response. Using original answer.",
                        title="Reasoning Validation Warning",
                        level=LogLevel.INFO,
                    )
                    return proposed_answer
                    
            except Exception as e:
                self.logger.log_markdown(
                    content=f"⚠️ Failed to parse XML response: {e}. Using original answer.",
                    title="Reasoning Validation XML Error",
                    level=LogLevel.INFO,
                )
                return proposed_answer
                
        except Exception as e:
            self.logger.log_markdown(
                content=f"❌ Error in reasoning validation: {e}. Using original answer.",
                title="Reasoning Validation Error",
                level=LogLevel.INFO,
            )
            return proposed_answer



    def _extract_final_answer_xml(self, content: str) -> Optional[str]:
        """
        Extract final_answer from XML-formatted content with error tolerance.
        
        Args:
            content (str): The content to parse
            
        Returns:
            Optional[str]: The extracted answer string, or None if not found
        """
        
        import re
        
        try:
            # Clean up the content
            cleaned_content = content.strip()
            
            # Define multiple patterns to handle various possible formats with error tolerance
            patterns = [
                r'<final_answer>(.*?)</final_answer>',
                r'<FINAL_ANSWER>(.*?)</FINAL_ANSWER>',
                r'<Final_Answer>(.*?)</Final_Answer>',
                r'<final_answer>(.*?)<final_answer>',
                r'<FINAL_ANSWER>(.*?)<FINAL_ANSWER>',
                r'<final_answer>(.*?)(?=<[^/]|$)',
                r'<FINAL_ANSWER>(.*?)(?=<[^/]|$)',
                r'<final_answer[^>]*>(.*?)</final_answer>',
                r'<FINAL_ANSWER[^>]*>(.*?)</FINAL_ANSWER>',
            ]
            
            # Try each pattern
            for pattern in patterns:
                match = re.search(pattern, cleaned_content, re.DOTALL | re.IGNORECASE)
                if match:
                    extracted_content = match.group(1).strip()
                    if extracted_content:  # Only return non-empty content
                        self.logger.log(
                            f"Successfully extracted final answer using pattern: {pattern[:30]}...",
                            level=LogLevel.DEBUG
                        )
                        return extracted_content
            
            # If no pattern matches, try a more lenient approach
            # Look for content between any tags that might be final_answer
            fallback_patterns = [
                r'<[^>]*final_answer[^>]*>(.*?)<[^>]*final_answer[^>]*>',
                r'<[^>]*final_answer[^>]*>(.*?)(?=<[^/]|$)',
            ]
            
            for pattern in fallback_patterns:
                match = re.search(pattern, cleaned_content, re.DOTALL | re.IGNORECASE)
                if match:
                    extracted_content = match.group(1).strip()
                    if extracted_content:
                        self.logger.log(
                            f"Extracted final answer using fallback pattern: {pattern[:30]}...",
                            level=LogLevel.DEBUG
                        )
                        return extracted_content
            
            # Final fallback: if content looks like it might be the answer without tags
            # Check if the content directly starts with the expected format
            if "### 1. Task outcome" in cleaned_content:
                self.logger.log(
                    "Content appears to contain answer format without XML tags",
                    level=LogLevel.DEBUG
                )
                return cleaned_content
            
            self.logger.log(
                "No final_answer XML tags found in content",
                level=LogLevel.DEBUG
            )
            return None
            
        except Exception as e:
            self.logger.log(f"XML parsing failed: {e}", level=LogLevel.DEBUG)
            return None

    def _build_reasoning_context(self) -> str:
        """
        Build reasoning context with complete reasoning process including tool calls and model outputs.
        """
        try:
            # Get memory from all completed steps
            agent_memory = self.write_memory_to_messages(summary_mode=True)
            
            # Also include the current step if it has tool calls and observations
            current_step = None
            if hasattr(self, '_current_memory_step') and self._current_memory_step:
                current_step = self._current_memory_step
                # Add current step's tool calls and observations to memory
                current_step_messages = current_step.to_messages(summary_mode=True)
                agent_memory.extend(current_step_messages)
            
            # Format the memory messages into a readable context
            context_parts = []
            context_parts.append(f"Original Task: {self.task}")
            context_parts.append("\n" + "="*50 + "\n")
            context_parts.append("=== AGENT MEMORY (complete reasoning process) ===")
            
            for i, message in enumerate(agent_memory):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                
                # Handle content based on its type
                if isinstance(content, list):
                    # Extract text from content list
                    text_content = ""
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_content += item.get("text", "")
                elif isinstance(content, str):
                    text_content = content
                else:
                    text_content = str(content)
                
                # Add each message to context
                context_parts.append(f"\nMessage {i+1} ({role}):")
                context_parts.append(text_content.strip())
                context_parts.append("-" * 40)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            # Fallback to basic task information
            self.logger.log(f"Warning: Failed to get agent memory: {e}", level=LogLevel.INFO)
            return f"Original Task: {self.task}\n\nError retrieving full context: {e}"


class CodeAgent(MultiStepAgent):
    """
    In this agent, the tool calls will be formulated by the LLM in code format, then parsed and executed.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        grammar (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
        additional_authorized_imports (`list[str]`, *optional*): Additional authorized imports for the agent.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        executor_type (`str`, default `"local"`): Which executor type to use between `"local"`, `"e2b"`, or `"docker"`.
        executor_kwargs (`dict`, *optional*): Additional arguments to pass to initialize the executor.
        max_print_outputs_length (`int`, *optional*): Maximum length of the print outputs.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        tools: List[Tool],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        prompt_templates: Optional[PromptTemplates] = None,
        grammar: Optional[Dict[str, str]] = None,
        additional_authorized_imports: Optional[List[str]] = None,
        planning_interval: Optional[int] = None,
        executor_type: str = "local",
        executor_kwargs: Optional[Dict[str, Any]] = None,
        max_print_outputs_length: Optional[int] = None,
        rethink_model: Optional[Callable[[List[Dict[str, str]]], ChatMessage]] = None,
        **kwargs,
    ):
        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        self.max_print_outputs_length = max_print_outputs_length
        self.rethink_model = rethink_model
        prompt_templates = prompt_templates or yaml.safe_load(
            importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text()
        )
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            grammar=grammar,
            planning_interval=planning_interval,
            **kwargs,
        )
        if "*" in self.additional_authorized_imports:
            self.logger.log(
                "Caution: you set an authorization for all imports, meaning your agent can decide to import any package it deems necessary. This might raise issues if the package is not installed in your environment.",
                0,
            )
        self.executor_type = executor_type
        self.executor_kwargs = executor_kwargs or {}
        self.python_executor = self.create_python_executor(executor_type, self.executor_kwargs)

    def create_python_executor(self, executor_type: str, kwargs: Dict[str, Any]) -> PythonExecutor:
        match executor_type:
            case "e2b" | "docker":
                if self.managed_agents:
                    raise Exception("Managed agents are not yet supported with remote code execution.")
                if executor_type == "e2b":
                    return E2BExecutor(self.additional_authorized_imports, self.logger, **kwargs)
                else:
                    return DockerExecutor(self.additional_authorized_imports, self.logger, **kwargs)
            case "local":
                return LocalPythonExecutor(
                    self.additional_authorized_imports,
                    max_print_outputs_length=self.max_print_outputs_length,
                )
            case _:  # if applicable
                raise ValueError(f"Unsupported executor type: {executor_type}")

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "authorized_imports": (
                    "You can import from any package you want."
                    if "*" in self.authorized_imports
                    else str(self.authorized_imports)
                ),
            },
        )
        return system_prompt

    def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        memory_messages = self.write_memory_to_messages()

        self.input_messages = memory_messages.copy()

        # Add new step in logs
        memory_step.model_input_messages = memory_messages.copy()

        # 打印API调用输入
        # print("\n[API调用] 执行步骤 - 发送给模型的输入:")
        # print(memory_step.model_input_messages)

        try:
            additional_args = {"grammar": self.grammar} if self.grammar is not None else {}
            chat_message: ChatMessage = self.model(
                self.input_messages,
                stop_sequences=["<end_code>", "Observation:"],
                **additional_args,
            )
            # 打印API调用输出
            # print("\n[API调用] 模型返回的输出:")
            # print(chat_message.content)
            
            memory_step.model_output_message = chat_message
            model_output = chat_message.content
            memory_step.model_output = model_output
        except Exception as e:
            raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

        self.logger.log_markdown(
            content=model_output,
            title="Output message of the LLM:",
            level=LogLevel.DEBUG,
        )

        # Parse
        try:
            code_action = fix_final_answer_code(parse_code_blobs(model_output))
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            raise AgentParsingError(error_msg, self.logger)

        memory_step.tool_calls = [
            ToolCall(
                name="python_interpreter",
                arguments=code_action,
                id=f"call_{len(self.memory.steps)}",
            )
        ]

        # Execute
        self.logger.log_code(title="Executing parsed code:", content=code_action, level=LogLevel.INFO)
        is_final_answer = False
        try:
            output, execution_logs, is_final_answer = self.python_executor(code_action)
            execution_outputs_console = []
            if len(execution_logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(execution_logs),
                ]
            observation = "Execution logs:\n" + execution_logs
        except Exception as e:
            if hasattr(self.python_executor, "state") and "_print_outputs" in self.python_executor.state:
                execution_logs = str(self.python_executor.state["_print_outputs"])
                if len(execution_logs) > 0:
                    execution_outputs_console = [
                        Text("Execution logs:", style="bold"),
                        Text(execution_logs),
                    ]
                    memory_step.observations = "Execution logs:\n" + execution_logs
                    self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.log(
                    "[bold red]Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your CodeAgent.",
                    level=LogLevel.INFO,
                )
            raise AgentExecutionError(error_msg, self.logger)

        truncated_output = truncate_content(str(output))
        observation += "Last output from code snippet:\n" + truncated_output
        memory_step.observations = observation

        execution_outputs_console += [
            Text(
                f"{('Out - Final answer' if is_final_answer else 'Out')}: {truncated_output}",
                style=(f"bold {YELLOW_HEX}" if is_final_answer else ""),
            ),
        ]
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        memory_step.action_output = output
        
        # Remove CodeAgent's own rethink logic to avoid conflict with MultiStepAgent's validation system
        # The MultiStepAgent's _execute_step method will handle validation and format checking
        return output if is_final_answer else None

    def _rethink_final_answer(self, proposed_answer: Any) -> Any:
        """
        Use the rethink model to reflect on the proposed final answer and correct it if needed.
        This process has two stages: reasoning validation and format checking.
        """
        if not self.rethink_model:
            return proposed_answer
            
        # Set flag to prevent recursive rethink calls
        self._in_rethink_process = True
        
        try:
            # Stage 1: Reasoning validation
            # reasoning_answer = self._rethink_reasoning_validation_code(proposed_answer)
            reasoning_answer = proposed_answer
            
            # If reasoning validation was skipped, initialize rethink_info with original answer
            if hasattr(self, '_current_memory_step') and self._current_memory_step:
                if not hasattr(self._current_memory_step, 'rethink_info'):
                    self._current_memory_step.rethink_info = {}
                # Record that reasoning was skipped and the original answer (only if not already set by reasoning)
                if 'original_answer' not in self._current_memory_step.rethink_info:
                    self._current_memory_step.rethink_info['original_answer'] = proposed_answer
                # Only mark as skipped if reasoning wasn't actually executed
                if 'reasoning_rethink_response' not in self._current_memory_step.rethink_info:
                    self._current_memory_step.rethink_info['reasoning_rethink_skipped'] = True
            
            # Stage 2: Format checking
            final_answer = self._rethink_format_checking_code(reasoning_answer)
            
            return final_answer
            
        except Exception as e:
            self.logger.log_markdown(
                content=f"❌ Error in rethink process: {e}. Using original answer.",
                title="Rethink Error",
                level=LogLevel.INFO,
            )
            return proposed_answer
            
        finally:
            # Always reset the flag when exiting rethink process
            self._in_rethink_process = False

    def _rethink_reasoning_validation_code(self, proposed_answer: Any) -> Any:
        """
        Stage 1: Use the rethink model to validate the reasoning process (CodeAgent version).
        """
        try:
            # Build the reasoning context from memory
            reasoning_context = self._build_reasoning_context()
            
            # Create the reasoning rethink prompt
            rethink_prompt = populate_template(
                self.prompt_templates["rethink"]["reasoning"],
                variables={
                    "task": self.task,
                    "reasoning_context": reasoning_context,
                    "proposed_answer": str(proposed_answer),
                },
            )
            
            # Prepare the input messages for the rethink model
            rethink_messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": rethink_prompt}],
                }
            ]
            
            self.logger.log_markdown(
                content="🤔 Initiating reasoning validation ...",
                title="Rethink Stage: Reasoning",
                level=LogLevel.INFO,
            )
            
            # Call the rethink model
            rethink_response = self.rethink_model(rethink_messages)
            rethink_content = rethink_response.content
            
            # Record rethink information in current memory step
            if hasattr(self, '_current_memory_step') and self._current_memory_step:
                if not hasattr(self._current_memory_step, 'rethink_info'):
                    self._current_memory_step.rethink_info = {}
                self._current_memory_step.rethink_info.update({
                    'reasoning_rethink_prompt': rethink_prompt,
                    'reasoning_rethink_input_messages': rethink_messages,
                    'reasoning_rethink_response': rethink_response,
                    'reasoning_rethink_content': rethink_content,
                    'original_answer': proposed_answer
                })
            
            self.logger.log_markdown(
                content=rethink_content,
                title="Reasoning Validation Response",
                level=LogLevel.INFO,
            )
            
            # Parse the rethink response using the same code parsing approach as CodeAgent
            try:
                # Parse code blocks from rethink response, similar to how CodeAgent does it
                code_action = fix_final_answer_code(parse_code_blobs(rethink_content))
                
                self.logger.log_code(
                    title="Executing reasoning validation code:", 
                    content=code_action, 
                    level=LogLevel.INFO
                )
                
                # Execute the rethink code using the python executor
                output, execution_logs, is_final_answer = self.python_executor(code_action)
                
                if execution_logs:
                    self.logger.log_markdown(
                        content=f"Reasoning validation execution logs:\n{execution_logs}",
                        title="Reasoning Validation Execution",
                        level=LogLevel.INFO,
                    )
                
                if is_final_answer:
                    # Record reasoning validation execution results
                    # Ensure rethink_info exists for execution results
                    if hasattr(self, '_current_memory_step') and self._current_memory_step:
                        if not hasattr(self._current_memory_step, 'rethink_info'):
                            self._current_memory_step.rethink_info = {}
                        self._current_memory_step.rethink_info.update({
                            'reasoning_rethink_code': code_action,
                            'reasoning_rethink_execution_logs': execution_logs,
                            'reasoning_rethink_final_answer': output,
                            'reasoning_rethink_success': True
                        })
                    
                    self.logger.log_markdown(
                        content=f"✅ Reasoning validation completed. Answer: {output}",
                        title="Reasoning Validation Result",
                        level=LogLevel.INFO,
                    )
                    return output
                else:
                    self.logger.log_markdown(
                        content="⚠️ Reasoning validation executed code but did not return a final answer. Using original answer.",
                        title="Reasoning Validation Warning",
                        level=LogLevel.INFO,
                    )
                    return proposed_answer
                    
            except Exception as e:
                self.logger.log_markdown(
                    content=f"⚠️ Failed to parse or execute reasoning validation code: {e}. Using original answer.",
                    title="Reasoning Validation Code Error",
                    level=LogLevel.INFO,
                )
                return proposed_answer
                
        except Exception as e:
            self.logger.log_markdown(
                content=f"❌ Error in reasoning validation: {e}. Using original answer.",
                title="Reasoning Validation Error",
                level=LogLevel.INFO,
            )
            return proposed_answer

    def _rethink_format_checking_code(self, proposed_answer: Any) -> Any:
        """
        Stage 2: Use the rethink model to check the format of the answer (CodeAgent version).
        """
        try:
            # Build reasoning context from memory (includes reasoning stage)
            reasoning_context = self._build_reasoning_context()
            
            # Create the format checking prompt with reasoning context
            format_prompt = populate_template(
                self.prompt_templates["rethink"]["format"],
                variables={
                    "task": self.task,
                    "reasoning_context": reasoning_context,
                    "proposed_answer": str(proposed_answer),
                },
            )
            
            # Prepare the input messages for the format checking
            format_messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": format_prompt}],
                }
            ]
            
            self.logger.log_markdown(
                content="📝 Initiating format checking ...",
                title="Rethink Stage: Format",
                level=LogLevel.INFO,
            )
            
            # Call the rethink model
            format_response = self.rethink_model(format_messages)
            format_content = format_response.content
            
            # Record format checking information in current memory step
            # Ensure rethink_info exists regardless of whether reasoning was executed
            if hasattr(self, '_current_memory_step') and self._current_memory_step:
                if not hasattr(self._current_memory_step, 'rethink_info'):
                    self._current_memory_step.rethink_info = {}
                self._current_memory_step.rethink_info.update({
                    'format_rethink_prompt': format_prompt,
                    'format_rethink_input_messages': format_messages,
                    'format_rethink_response': format_response,
                    'format_rethink_content': format_content,
                })
            
            self.logger.log_markdown(
                content=format_content,
                title="Format Checking Response",
                level=LogLevel.INFO,
            )
            
            # Parse the format checking response using the same code parsing approach as CodeAgent
            try:
                # Parse code blocks from format checking response
                code_action = fix_final_answer_code(parse_code_blobs(format_content))
                
                self.logger.log_code(
                    title="Executing format checking code:", 
                    content=code_action, 
                    level=LogLevel.INFO
                )
                
                # Execute the format checking code using the python executor
                output, execution_logs, is_final_answer = self.python_executor(code_action)
                
                if execution_logs:
                    self.logger.log_markdown(
                        content=f"Format checking execution logs:\n{execution_logs}",
                        title="Format Checking Execution",
                        level=LogLevel.INFO,
                    )
                
                if is_final_answer:
                    # Record format checking execution results
                    # Ensure rethink_info exists for execution results
                    if hasattr(self, '_current_memory_step') and self._current_memory_step:
                        if not hasattr(self._current_memory_step, 'rethink_info'):
                            self._current_memory_step.rethink_info = {}
                        self._current_memory_step.rethink_info.update({
                            'format_rethink_code': code_action,
                            'format_rethink_execution_logs': execution_logs,
                            'format_rethink_final_answer': output,
                            'format_rethink_success': True
                        })
                    
                    self.logger.log_markdown(
                        content=f"✅ Format checking completed. Final answer: {output}",
                        title="Format Checking Result",
                        level=LogLevel.INFO,
                    )
                    return output
                else:
                    self.logger.log_markdown(
                        content="⚠️ Format checking executed code but did not return a final answer. Using input answer.",
                        title="Format Checking Warning",
                        level=LogLevel.INFO,
                    )
                    return proposed_answer
                    
            except Exception as e:
                self.logger.log_markdown(
                    content=f"⚠️ Failed to parse or execute format checking code: {e}. Using input answer.",
                    title="Format Checking Code Error",
                    level=LogLevel.INFO,
                )
                return proposed_answer
                
        except Exception as e:
            self.logger.log_markdown(
                content=f"❌ Error in format checking: {e}. Using input answer.",
                title="Format Checking Error",
                level=LogLevel.INFO,
            )
            return proposed_answer

    def _build_reasoning_context(self) -> str:
        """
        Build reasoning context with complete reasoning process including tool calls and model outputs.
        """
        try:
            # Get memory from all completed steps
            agent_memory = self.write_memory_to_messages(summary_mode=True)
            
            # Also include the current step if it has tool calls and observations
            current_step = None
            if hasattr(self, '_current_memory_step') and self._current_memory_step:
                current_step = self._current_memory_step
                # Add current step's tool calls and observations to memory
                current_step_messages = current_step.to_messages(summary_mode=True)
                agent_memory.extend(current_step_messages)
            

            
            # Format the memory messages into a readable context
            context_parts = []
            context_parts.append(f"Original Task: {self.task}")
            context_parts.append("\n" + "="*50 + "\n")
            context_parts.append("=== AGENT MEMORY (complete reasoning process) ===")
            
            for i, message in enumerate(agent_memory):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                
                # Handle content based on its type
                if isinstance(content, list):
                    # Extract text from content list
                    text_content = ""
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_content += item.get("text", "")
                elif isinstance(content, str):
                    text_content = content
                else:
                    text_content = str(content)
                
                # Add each message to context
                context_parts.append(f"\nMessage {i+1} ({role}):")
                context_parts.append(text_content.strip())
                context_parts.append("-" * 40)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            # Fallback to basic task information
            self.logger.log(f"Warning: Failed to get agent memory: {e}", level=LogLevel.INFO)
            return f"Original Task: {self.task}\n\nError retrieving full context: {e}"

    def _mark_validation_steps(self):
        """
        Mark memory steps that should be included in validation.
        Only includes step 0 (original task) and python_interpreter calls (except final_answer).
        """
        for memory_step in self.memory.steps:
            if hasattr(memory_step, 'include_in_validation'):
                # Check if this is a python_interpreter call
                if hasattr(memory_step, 'tool_calls') and memory_step.tool_calls:
                    for tool_call in memory_step.tool_calls:
                        if tool_call.name == "python_interpreter":
                            # Check if this is NOT a final_answer call
                            arguments = tool_call.arguments
                            if isinstance(arguments, str) and "final_answer" not in arguments.lower():
                                memory_step.include_in_validation = True
                                break

    def _handle_max_steps_reached(self, task: str, images: List[str], step_start_time: float) -> Any:
        """
        CodeAgent-specific handler for when max steps are reached.
        
        For CodeAgent:
        1. Provide final answer based on current memory
        2. Apply format checking if available
        3. Create separate memory step for format checking process
        4. Return enhanced answer
        """
        # Mark validation as completed and set flags to skip validation step
        if self.validation_agent and self.original_problem:
            self._validation_completed = True
            self.first_final_answer_called = True
        
        # Mark this as a max steps situation for format checking
        self._is_max_steps_reached = True
            
        # Step 1: Provide final answer based on current memory
        final_answer = self.provide_final_answer(task, images)
        
        # Step 2: Create final memory step with max steps error and original answer
        final_memory_step = ActionStep(
            step_number=self.step_number, error=AgentMaxStepsError("Reached max steps.", self.logger)
        )
        final_memory_step.action_output = final_answer
        final_memory_step.end_time = time.time()
        final_memory_step.duration = final_memory_step.end_time - step_start_time
        self.memory.steps.append(final_memory_step)
        
        # Step 3: Run callbacks for the original answer
        for callback in self.step_callbacks:
            callback(final_memory_step) if len(inspect.signature(callback).parameters) == 1 else callback(
                final_memory_step, agent=self
            )
        
        # Step 4: Apply format checking if available and create separate memory step
        if hasattr(self, 'rethink_model') and self.rethink_model:
            try:
                # Create a new memory step for format checking process
                format_check_step_start_time = time.time()
                format_check_memory_step = ActionStep(
                    step_number=self.step_number + 1,  # Next step number
                    start_time=format_check_step_start_time
                )
                
                # Set up the memory step to match normal format checking flow
                format_check_memory_step.model_output = """Thought: Apply format checking to improve the final answer.
Code:
```py
# Format checking process for max steps reached scenario
format_checking_result = _rethink_format_checking_code(final_answer)
```"""
                
                # Create a mock tool call for format checking (similar to python_interpreter)
                from smolagents.memory import ToolCall
                format_check_tool_call = ToolCall(
                    id=f"call_{len(self.memory.steps)}",
                    name="python_interpreter",
                    arguments="format_checking_result = _rethink_format_checking_code(final_answer)"
                )
                format_check_memory_step.tool_calls = [format_check_tool_call]
                
                # Use format checking rethink to enhance the final answer (same as normal flow)
                enhanced_answer = self._rethink_format_checking_code(final_answer)
                
                # Record format checking observation (similar to normal flow)
                format_check_observation = f"""Execution logs:
Format checking process for max steps scenario completed.
Last output from code snippet:
{enhanced_answer}"""
                format_check_memory_step.observations = format_check_observation
                format_check_memory_step.action_output = enhanced_answer
                
                # Finalize the format checking memory step (same as _finalize_step)
                format_check_memory_step.end_time = time.time()
                format_check_memory_step.duration = format_check_memory_step.end_time - format_check_step_start_time
                self.memory.steps.append(format_check_memory_step)
                
                # Run callbacks for the format checking step (same as normal flow)
                for callback in self.step_callbacks:
                    callback(format_check_memory_step) if len(inspect.signature(callback).parameters) == 1 else callback(
                        format_check_memory_step, agent=self
                    )
                
                self.logger.log_markdown(
                    content=f"✅ Max steps reached - Format checking completed and recorded in memory.",
                    title="Max Steps - Format Checking Applied",
                    level=LogLevel.INFO,
                )
                
                return enhanced_answer
                
            except Exception as e:
                self.logger.log_markdown(
                    content=f"⚠️ Format checking failed at max steps: {e}. Using original final answer.",
                    title="Max Steps - Format Checking Error",
                    level=LogLevel.INFO,
                )
            finally:
                # Clean up max steps flag
                self._is_max_steps_reached = False
            
        return final_answer


class SimpleCoder(MultiStepAgent):
    """
    A simplified code agent focused on quick code generation and execution.
    
    This agent is designed to:
    - Handle simple coding tasks efficiently (max 3 steps)
    - Generate and execute code without complex planning
    - Provide quick results for computational tasks
    - Automatically retrieve relevant API endpoints based on task
    
    Args:
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        tools (`list[Tool]`, *optional*): Additional tools for the agent. Defaults to empty list.
        additional_authorized_imports (`list[str]`, *optional*): Additional authorized imports for the agent.
        executor_type (`str`, default `"local"`): Which executor type to use between `"local"`, `"e2b"`, or `"docker"`.
        executor_kwargs (`dict`, *optional*): Additional arguments to pass to initialize the executor.
        max_print_outputs_length (`int`, *optional*): Maximum length of the print outputs.
        enable_api_retrieval (`bool`, default `True`): Whether to enable automatic API endpoint retrieval.
        api_db_paths (`list[str]`, *optional*): Paths to API database JSON files.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        tools: Optional[List[Tool]] = None,
        additional_authorized_imports: Optional[List[str]] = None,
        executor_type: str = "local",
        executor_kwargs: Optional[Dict[str, Any]] = None,
        max_print_outputs_length: Optional[int] = None,
        enable_api_retrieval: bool = True,
        api_db_paths: Optional[List[str]] = None,
        **kwargs,
    ):
        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        self.max_print_outputs_length = max_print_outputs_length
        
        # API retrieval settings
        self.enable_api_retrieval = enable_api_retrieval
        self.api_retriever = None
        self._api_context = ""
        self._auth_info = ""
        
        # Initialize API retriever if enabled
        if enable_api_retrieval:
            self._init_api_retriever(api_db_paths)
        
        # Load SimpleCoder specific prompt templates
        prompt_templates = yaml.safe_load(
            importlib.resources.files("smolagents.prompts").joinpath("simple_coder_agent.yaml").read_text()
        )
        
        super().__init__(
            tools=tools or [],
            model=model,
            prompt_templates=prompt_templates,
            max_steps=5,  # Maximum 5 steps for SimpleCoder
            planning_interval=None,  # No planning for SimpleCoder
            **kwargs,
        )
        
        if "*" in self.additional_authorized_imports:
            self.logger.log(
                "Caution: you set an authorization for all imports, meaning your agent can decide to import any package it deems necessary. This might raise issues if the package is not installed in your environment.",
                0,
            )
        self.executor_type = executor_type
        self.executor_kwargs = executor_kwargs or {}
        self.python_executor = self.create_python_executor(executor_type, self.executor_kwargs)
        
        # Inject environment variables into executor state
        self._inject_api_keys()
    
    def _init_api_retriever(self, api_db_paths: Optional[List[str]] = None):
        """Initialize the API retriever for endpoint search."""
        try:
            # Try to import the API retriever from the expected location
            import sys
            from pathlib import Path
            
            # Add the scripts directory to path if needed
            scripts_path = Path(__file__).parent.parent.parent / "examples" / "open_deep_research" / "scripts"
            if scripts_path.exists() and str(scripts_path) not in sys.path:
                sys.path.insert(0, str(scripts_path))
            
            from api_retrieval import APIRetriever
            self.api_retriever = APIRetriever(api_db_paths)
            logger.info(f"API retriever initialized with {len(self.api_retriever.endpoints)} endpoints")
        except ImportError as e:
            logger.warning(f"Could not initialize API retriever: {e}")
            self.api_retriever = None
        except Exception as e:
            logger.warning(f"Error initializing API retriever: {e}")
            self.api_retriever = None
    
    def _inject_api_keys(self):
        """Inject API keys from environment variables into the executor state."""
        # Get API keys from environment variables
        api_keys = {}
        
        # YouTube API Key
        youtube_key = os.environ.get("YOUTUBE_API_KEY")
        if youtube_key:
            api_keys["YOUTUBE_API_KEY"] = youtube_key
        
        # ORCID Token (if available)
        orcid_token = os.environ.get("ORCID_ACCESS_TOKEN")
        if orcid_token:
            api_keys["ORCID_ACCESS_TOKEN"] = orcid_token
        
        # Inject into executor state
        if api_keys and hasattr(self, 'python_executor'):
            self.python_executor.send_variables(api_keys)
    
    def _retrieve_api_context(self, task: str) -> None:
        """Retrieve relevant API endpoints for the given task."""
        if not self.api_retriever:
            self._api_context = ""
            self._auth_info = ""
            return
        
        try:
            # Search for relevant endpoints
            endpoints = self.api_retriever.search(task, top_k=3)
            
            if endpoints:
                # Generate API context
                self._api_context = self.api_retriever.generate_api_context(endpoints)
                
                # Check authentication requirements and generate auth info
                auth_reqs = self.api_retriever.get_auth_requirements(endpoints)
                auth_info_lines = []
                
                if auth_reqs.get("youtube_api_key"):
                    youtube_key = os.environ.get("YOUTUBE_API_KEY")
                    if youtube_key:
                        auth_info_lines.append(
                            "- **YouTube API Key:** Available as `YOUTUBE_API_KEY` variable in your code. "
                            "Use it directly: `key = YOUTUBE_API_KEY`"
                        )
                    else:
                        auth_info_lines.append(
                            "- **YouTube API Key:** Required but not configured. "
                            "Set the YOUTUBE_API_KEY environment variable."
                        )
                
                if auth_reqs.get("orcid_token"):
                    orcid_token = os.environ.get("ORCID_ACCESS_TOKEN")
                    if orcid_token:
                        auth_info_lines.append(
                            "- **ORCID Access Token:** Available as `ORCID_ACCESS_TOKEN` variable in your code."
                        )
                    else:
                        auth_info_lines.append(
                            "- **ORCID Access Token:** May be required for some endpoints. "
                            "Public API endpoints work without authentication."
                        )
                
                self._auth_info = "\n".join(auth_info_lines) if auth_info_lines else ""
            else:
                self._api_context = ""
                self._auth_info = ""
                
        except Exception as e:
            logger.warning(f"Error retrieving API context: {e}")
            self._api_context = ""
            self._auth_info = ""

    def create_python_executor(self, executor_type: str, kwargs: Dict[str, Any]) -> PythonExecutor:
        match executor_type:
            case "e2b" | "docker":
                if self.managed_agents:
                    raise Exception("Managed agents are not yet supported with remote code execution.")
                if executor_type == "e2b":
                    return E2BExecutor(self.additional_authorized_imports, self.logger, **kwargs)
                else:
                    return DockerExecutor(self.additional_authorized_imports, self.logger, **kwargs)
            case "local":
                return LocalPythonExecutor(
                    self.additional_authorized_imports,
                    max_print_outputs_length=self.max_print_outputs_length,
                )
            case _:
                raise ValueError(f"Unsupported executor type: {executor_type}")

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "authorized_imports": (
                    "You can import from any package you want."
                    if "*" in self.authorized_imports
                    else str(self.authorized_imports)
                ),
            },
        )
        return system_prompt
    
    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: Optional[List[str]] = None,
        additional_args: Optional[Dict] = None,
        max_steps: Optional[int] = None,
    ):
        """
        Run the SimpleCoder agent for the given task.
        
        Overrides the parent run method to:
        1. Retrieve relevant API endpoints before execution
        2. Inject API context into the task prompt
        3. Re-inject API keys into executor state on reset
        
        Args:
            task (`str`): Task to perform.
            stream (`bool`): Whether to run in a streaming way.
            reset (`bool`): Whether to reset the conversation or keep it going from previous run.
            images (`list[str]`, *optional*): Paths to image(s).
            additional_args (`dict`, *optional*): Additional variables to pass to the agent.
            max_steps (`int`, *optional*): Maximum number of steps.
        """
        # Re-inject API keys on reset (in case executor state was cleared)
        if reset:
            self._inject_api_keys()
        
        # Retrieve API context based on task
        if self.enable_api_retrieval:
            self._retrieve_api_context(task)
        
        # Call parent run method
        return super().run(
            task=task,
            stream=stream,
            reset=reset,
            images=images,
            additional_args=additional_args,
            max_steps=max_steps,
        )
    
    def __call__(self, task: str, **kwargs):
        """
        Adds additional prompting for the managed agent, runs it, and wraps the output.
        This method is called only when SimpleCoder is used as a managed agent.
        
        Overrides the parent __call__ method to inject API context into the task prompt.
        """
        # Get original problem from self or from main agent
        original_problem = getattr(self, 'original_problem', None) or ""
        if not original_problem and hasattr(self, '_main_agent') and self._main_agent:
            original_problem = getattr(self._main_agent, 'original_problem', '') or ""
        
        # Retrieve API context based on task
        if self.enable_api_retrieval:
            self._retrieve_api_context(task)
        
        # Build the full task with API context
        full_task = populate_template(
            self.prompt_templates["managed_agent"]["task"],
            variables=dict(
                name=self.name, 
                task=task, 
                problem=original_problem,
                api_context=self._api_context,
                auth_info=self._auth_info,
            ),
        )
        
        # Preserve the current original_problem to avoid it being overwritten by reset=True
        preserved_original_problem = self.original_problem
        
        report = self.run(full_task, reset=True, **kwargs)
        
        # Restore the original_problem after reset
        self.original_problem = preserved_original_problem
        
        answer = populate_template(
            self.prompt_templates["managed_agent"]["report"], 
            variables=dict(name=self.name, final_answer=report)
        )
        return answer

    def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        memory_messages = self.write_memory_to_messages()
        self.input_messages = memory_messages.copy()
        memory_step.model_input_messages = memory_messages.copy()

        try:
            chat_message: ChatMessage = self.model(
                self.input_messages,
                stop_sequences=["<end_code>", "Observation:"],
            )
            memory_step.model_output_message = chat_message
            model_output = chat_message.content
            memory_step.model_output = model_output
        except Exception as e:
            raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

        self.logger.log_markdown(
            content=model_output,
            title="SimpleCoder Output:",
            level=LogLevel.DEBUG,
        )

        # Parse
        try:
            code_action = fix_final_answer_code(parse_code_blobs(model_output))
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            raise AgentParsingError(error_msg, self.logger)

        memory_step.tool_calls = [
            ToolCall(
                name="python_interpreter",
                arguments=code_action,
                id=f"call_{len(self.memory.steps)}",
            )
        ]

        # Execute
        self.logger.log_code(title="Executing code:", content=code_action, level=LogLevel.INFO)
        is_final_answer = False
        try:
            output, execution_logs, is_final_answer = self.python_executor(code_action)
            execution_outputs_console = []
            if len(execution_logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(execution_logs),
                ]
            observation = "Execution logs:\n" + execution_logs
        except Exception as e:
            if hasattr(self.python_executor, "state") and "_print_outputs" in self.python_executor.state:
                execution_logs = str(self.python_executor.state["_print_outputs"])
                if len(execution_logs) > 0:
                    execution_outputs_console = [
                        Text("Execution logs:", style="bold"),
                        Text(execution_logs),
                    ]
                    memory_step.observations = "Execution logs:\n" + execution_logs
                    self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.log(
                    "[bold red]Warning: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your SimpleCoder.",
                    level=LogLevel.INFO,
                )
            raise AgentExecutionError(error_msg, self.logger)

        truncated_output = truncate_content(str(output))
        observation += "Last output from code snippet:\n" + truncated_output
        memory_step.observations = observation

        execution_outputs_console += [
            Text(
                f"{('Out - Final answer' if is_final_answer else 'Out')}: {truncated_output}",
                style=(f"bold {YELLOW_HEX}" if is_final_answer else ""),
            ),
        ]
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        memory_step.action_output = output
        return output if is_final_answer else None


class ValidationAgent:
    """
    Validation agent for two-stage verification of agent reasoning.
    
    Stage 1: Organize proof - Restructure reasoning steps in proof format
    Stage 2: Verify each step - Validate each reasoning step individually
    """

    def __init__(self, model):
        """
        Initialize ValidationAgent with a model instance.
        
        Args:
            model: The model instance to use for validation
        """
        self.model = model
        self.logger = AgentLogger(level=LogLevel.INFO)
        
        # Load validation prompts
        import importlib.resources
        self.prompt_data = yaml.safe_load(
            importlib.resources.files("smolagents.prompts").joinpath("validation_agent_prompt.yaml").read_text()
        )

    def validate_answer(self, manager_agent, proposed_answer: str) -> str:
        """
        Main validation method that performs two-stage verification.
        
        Args:
            manager_agent: The manager agent instance
            proposed_answer: The answer to validate
            
        Returns:
            Validation feedback message for the manager agent
        """
        self.logger.log(
            f"─Stage 1 - Organizing proof:\n{'─' * 95}",
            level=LogLevel.INFO
        )
        
        # Stage 1: Organize proof
        validation_messages = self._extract_validation_messages(manager_agent, proposed_answer)
        # Store validation_messages for use in verification stage
        self.validation_messages = validation_messages
        self.original_problem = manager_agent.original_problem
        
        organized_proof = self._organize_proof(validation_messages, proposed_answer)
        
        self.logger.log(
            f"─Stage 1 Result - Organized proof:\n{'─' * 95}\n{organized_proof}",
            level=LogLevel.INFO
        )
        self.logger.log(
            f"─Stage 2 - Verifying reasoning steps:\n{'─' * 95}",
            level=LogLevel.INFO
        )
        
        # Stage 2: Verify each reasoning step
        feedback = self._verify_reasoning_steps(organized_proof, manager_agent.original_problem)
        
        self.logger.log(
            f"─Validation Final Result:\n{'─' * 95}\n{feedback}",
            level=LogLevel.INFO
        )
        
        return feedback

    def _extract_validation_messages(self, manager_agent, proposed_answer: str) -> List[Dict[str, str]]:
        """
        Extract validation messages from manager agent memory.
        Only includes step 0 (original task) and python_interpreter calls (except final_answer).
        Also adds final answer at the end.
        """
        validation_messages = []
        step_counter = 1
        
        # Add step 0: original task description
        validation_messages.append({
            "text": f"[step 0]{manager_agent.original_problem}",
            "step_number": 0
        })
        
        # Process memory steps that are marked for validation
        for memory_step in manager_agent.memory.steps:
            if hasattr(memory_step, 'include_in_validation') and memory_step.include_in_validation:
                if hasattr(memory_step, 'tool_calls') and memory_step.tool_calls:
                    for tool_call in memory_step.tool_calls:
                        if tool_call.name == "python_interpreter":
                            # Add tool call message
                            validation_messages.append({
                                "text": f"[step {step_counter}]Calling tools:\n{[tool_call.dict()]}",
                                "step_number": step_counter
                            })
                            
                            # Add observation message if available
                            if memory_step.observations:
                                validation_messages.append({
                                    "text": f"[step {step_counter}]Call id: {tool_call.id}\nObservation:\n{memory_step.observations}",
                                    "step_number": step_counter
                                })
                            
                            step_counter += 1
        
        # Add final answer using the provided proposed_answer
        validation_messages.append({
            "text": f"Final answer: {proposed_answer}",
            "step_number": "final"
        })
        
        return validation_messages

    def _organize_proof(self, validation_messages: List[Dict[str, str]], proposed_answer: str) -> str:
        """
        Stage 1: Organize reasoning steps in proof format.
        """
        from jinja2 import Template
        
        # Prepare prompt templates
        system_template = Template(self.prompt_data['validation']['system'])
        user_pre_template = Template(self.prompt_data['validation']['organize_proof_pre_messages'])
        user_post_template = Template(self.prompt_data['validation']['organize_proof_post_messages'])
        
        system_prompt = system_template.render()
        user_pre_prompt = user_pre_template.render()
        user_post_prompt = user_post_template.render()
        
        # Build messages
        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": MessageRole.USER,
                "content": [{"type": "text", "text": user_pre_prompt}],
            },
        ]
        
        # Add validation messages
        for msg in validation_messages:
            messages.append({
                "role": MessageRole.USER,
                "content": [{"type": "text", "text": msg["text"]}]
            })
        
        messages.append({
            "role": MessageRole.USER,
            "content": [{"type": "text", "text": user_post_prompt}],
        })
        
        response = self.model(messages)
        
        self.logger.log(
            f"─Stage 1 Model Response:\n{'─' * 95}\n{response.content}",
            level=LogLevel.INFO
        )
        
        return response.content

    def _verify_reasoning_steps(self, organized_proof: str, original_task: str) -> str:
        """
        Stage 2: Verify each reasoning step individually.
        """
        # Extract reasoning steps from organized proof
        reasoning_steps = self._extract_reasoning_steps(organized_proof)
        
        if not reasoning_steps:
            return "No reasoning steps found in the organized proof."
        
        suggestions = []
        
        # Verify each step except the last one (intermediate reasoning)
        for i, step in enumerate(reasoning_steps[:-1]):
            is_valid, suggestion = self._check_intermediate_reasoning(step, reasoning_steps[:i])
            if not is_valid and suggestion != "No suggestions":
                suggestions.append(suggestion)
        
        # Verify the last step (final reasoning)
        if len(reasoning_steps) > 0:
            last_step = reasoning_steps[-1]
            is_valid, suggestion = self._check_final_reasoning(last_step, reasoning_steps[:-1], original_task)
            if not is_valid and suggestion != "No suggestions":
                suggestions.append(suggestion)
        
        # Prepare feedback
        if not suggestions:
            return "All reasoning steps are correct."
        else:
            feedback = "Some reasoning steps need improvement:\n"
            for suggestion in suggestions:
                feedback += f"- {suggestion}\n"
            return feedback.strip()

    def _extract_reasoning_steps(self, organized_proof: str) -> List[Dict[str, str]]:
        """
        Extract individual reasoning steps from the organized proof with enhanced error tolerance.
        """
        # Clean up the organized proof
        cleaned_proof = organized_proof.strip()
        
        # Find all reasoning steps using multiple patterns for better tolerance
        # Handle various cases: [reasoning 1], [REASONING 1], spaces, etc.
        patterns = [
            r'\[reasoning\s+(\d+)\]\s*:?\s*(.*?)(?=\[reasoning\s+\d+\]|$)',
            r'\[REASONING\s+(\d+)\]\s*:?\s*(.*?)(?=\[REASONING\s+\d+\]|$)',
            r'\[reasoning\s+(\d+)\]\s*:?\s*(.*?)(?=\[REASONING\s+\d+\]|$)',
            r'\[REASONING\s+(\d+)\]\s*:?\s*(.*?)(?=\[reasoning\s+\d+\]|$)',
        ]
        
        all_matches = []
        for pattern in patterns:
            matches = re.findall(pattern, cleaned_proof, re.DOTALL | re.IGNORECASE)
            all_matches.extend(matches)
        
        # Remove duplicates and sort by step number
        unique_matches = {}
        for match in all_matches:
            step_num = int(match[0])
            if step_num not in unique_matches:
                unique_matches[step_num] = match[1].strip()
        
        steps = []
        for step_num in sorted(unique_matches.keys()):
            content = unique_matches[step_num]
            
            # Extract references with enhanced tolerance
            # Use a single comprehensive pattern with case-insensitive matching
            ref_pattern = r'<ref[^>]*>(.*?)(?:</ref>|(?=<[^/])|$)'
            found_refs = re.findall(ref_pattern, content, re.DOTALL | re.IGNORECASE)
            
            # Clean up references - remove quotes and extra spaces
            all_references = []
            for ref in found_refs:
                ref_cleaned = re.sub(r'^["\']|["\']$', '', ref.strip())
                if ref_cleaned:
                    # Split by semicolon and clean up each part
                    split_refs = [r.strip() for r in ref_cleaned.split(';') if r.strip()]
                    all_references.extend(split_refs)
            
            # Remove duplicates while preserving order
            unique_references = []
            seen = set()
            for ref in all_references:
                if ref not in seen:
                    unique_references.append(ref)
                    seen.add(ref)
            
            # Remove reference tags from content
            clean_content = re.sub(r'<ref[^>]*>.*?(?:</ref>|(?=<[^/])|$)', '', content, flags=re.DOTALL | re.IGNORECASE)
            clean_content = clean_content.strip()
            
            steps.append({
                "step_number": step_num,
                "content": clean_content,
                "references": unique_references
            })
        
        return steps

    def _check_intermediate_reasoning(self, step: Dict[str, str], previous_steps: List[Dict[str, str]]) -> Tuple[bool, str]:
        """
        Check intermediate reasoning step.
        """
        from jinja2 import Template
        
        # Prepare conditions from references
        conditions = []
        condition_counter = 1
        
        for ref in step["references"]:
            ref = ref.strip()
            if ref.startswith("step "):
                # Extract step number and find corresponding validation message content
                try:
                    step_num = int(ref.split()[1])
                    step_content = self._get_step_content_from_validation_messages(step_num)
                    if step_content:
                        conditions.append(f"Condition {condition_counter}:\n{step_content}")
                        condition_counter += 1
                except (ValueError, IndexError):
                    pass
            elif ref.startswith("reasoning "):
                # Extract reasoning number and find corresponding content from previous steps
                try:
                    reasoning_num = int(ref.split()[1]) - 1
                    if reasoning_num < len(previous_steps):
                        conditions.append(f"Condition {condition_counter}:\n{previous_steps[reasoning_num]['content']}")
                        condition_counter += 1
                except (ValueError, IndexError):
                    pass
        
        # Prepare prompt
        template = Template(self.prompt_data['validation']['check_intermediate_reasoning'])
        prompt = template.render(
            conditions="\n".join(conditions),
            inference=step["content"]
        )
        
        # Get validation response
        messages = [
            {
                "role": MessageRole.USER,
                "content": [{"type": "text", "text": prompt}]
            }
        ]
        
        response = self.model(messages).content
        
        self.logger.log(
            f"─Stage 2 Intermediate Check Response:\n{'─' * 95}\n{response}",
            level=LogLevel.INFO
        )
        
        return self._parse_validation_response(response)

    def _check_final_reasoning(self, step: Dict[str, str], previous_steps: List[Dict[str, str]], task: str) -> Tuple[bool, str]:
        """
        Check final reasoning step.
        """
        from jinja2 import Template
        
        # Prepare conditions from references
        conditions = []
        condition_counter = 1
        
        for ref in step["references"]:
            ref = ref.strip()
            if ref.startswith("step "):
                # Extract step number and find corresponding validation message content
                try:
                    step_num = int(ref.split()[1])
                    step_content = self._get_step_content_from_validation_messages(step_num)
                    if step_content:
                        conditions.append(f"Condition {condition_counter}:\n{step_content}")
                        condition_counter += 1
                except (ValueError, IndexError):
                    pass
            elif ref.startswith("reasoning "):
                # Extract reasoning number and find corresponding content from previous steps
                try:
                    reasoning_num = int(ref.split()[1]) - 1
                    if reasoning_num < len(previous_steps):
                        conditions.append(f"Condition {condition_counter}:\n{previous_steps[reasoning_num]['content']}")
                        condition_counter += 1
                except (ValueError, IndexError):
                    pass
        
        # Prepare prompt
        template = Template(self.prompt_data['validation']['check_final_reasoning'])
        prompt = template.render(
            task=task,
            conditions="\n".join(conditions),
            inference=step["content"]
        )
        
        # Get validation response
        messages = [
            {
                "role": MessageRole.USER,
                "content": [{"type": "text", "text": prompt}]
            }
        ]
        
        response = self.model(messages).content
        
        self.logger.log(
            f"─Stage 2 Final Check Response:\n{'─' * 95}\n{response}",
            level=LogLevel.INFO
        )
        
        return self._parse_validation_response(response)

    def _get_step_content_from_validation_messages(self, step_num: int) -> str:
        """
        Get step content from validation messages for the given step number.
        Returns the clean step content without [step n] prefix.
        For tool calls, combines both the call and its observation as a single content.
        """
        if not hasattr(self, 'validation_messages') or not self.validation_messages:
            return ""
        
        step_content_parts = []
        
        # Find all messages related to this step number
        for msg in self.validation_messages:
            if msg.get("step_number") == step_num:
                text = msg["text"]
                # Remove pattern like "[step n]" from the beginning
                cleaned_text = re.sub(r'^\[step\s+\d+\]', '', text).strip()
                if cleaned_text:
                    step_content_parts.append(cleaned_text)
        
        # Join all parts related to this step as a single content
        # This ensures tool calls and their observations are combined
        if step_content_parts:
            return "\n".join(step_content_parts)
        else:
            return ""

    def _parse_validation_response(self, response: str) -> Tuple[bool, str]:
        """
        Parse XML-formatted validation response with enhanced error tolerance.
        """
        # Clean up response for better parsing
        cleaned_response = response.strip()
        
        # Extract judgment with enhanced tolerance
        # Handle various cases: <judgment>, <JUDGMENT>, spaces, quotes, etc.
        judgment_patterns = [
            r'<judgment[^>]*>(.*?)</judgment>',
            r'<JUDGMENT[^>]*>(.*?)</JUDGMENT>',
            r'<judgment[^>]*>(.*?)(?=<[^/]|$)',  # Handle missing closing tag
            r'<JUDGMENT[^>]*>(.*?)(?=<[^/]|$)',
        ]
        
        judgment = "false"
        for pattern in judgment_patterns:
            judgment_match = re.search(pattern, cleaned_response, re.DOTALL | re.IGNORECASE)
            if judgment_match:
                judgment_raw = judgment_match.group(1).strip()
                # Remove quotes and clean up
                judgment_raw = re.sub(r'^["\']|["\']$', '', judgment_raw).strip()
                judgment = judgment_raw.lower()
                break
        
        # Extract suggestions with enhanced tolerance
        suggestions_patterns = [
            r'<suggestions[^>]*>(.*?)</suggestions>',
            r'<SUGGESTIONS[^>]*>(.*?)</SUGGESTIONS>',
            r'<suggestions[^>]*>(.*?)(?=<[^/]|$)',  # Handle missing closing tag
            r'<SUGGESTIONS[^>]*>(.*?)(?=<[^/]|$)',
        ]
        
        suggestions = "No suggestions"
        for pattern in suggestions_patterns:
            suggestions_match = re.search(pattern, cleaned_response, re.DOTALL | re.IGNORECASE)
            if suggestions_match:
                suggestions_raw = suggestions_match.group(1).strip()
                # Remove quotes and clean up
                suggestions_raw = re.sub(r'^["\']|["\']$', '', suggestions_raw).strip()
                if suggestions_raw:
                    suggestions = suggestions_raw
                break
        
        # Handle various true/false formats
        true_patterns = [
            r'^true$', r'^"true"$', r'^\'true\'$',
            r'^True$', r'^"True"$', r'^\'True\'$',
            r'^TRUE$', r'^"TRUE"$', r'^\'TRUE\'$'
        ]
        
        is_valid = any(re.match(pattern, judgment.strip(), re.IGNORECASE) for pattern in true_patterns)
        
        return is_valid, suggestions

