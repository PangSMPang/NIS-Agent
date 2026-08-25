from dataclasses import asdict, dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, List, TypedDict, Union

from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import AgentLogger, LogLevel
from smolagents.utils import AgentError, AgentMaxStepsError, make_json_serializable


if TYPE_CHECKING:
    from smolagents.models import ChatMessage
    from smolagents.monitoring import AgentLogger


logger = getLogger(__name__)


class Message(TypedDict):
    role: MessageRole
    content: str | list[dict]


@dataclass
class ToolCall:
    name: str
    arguments: Any
    id: str

    def dict(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": make_json_serializable(self.arguments),
            },
        }


@dataclass
class MemoryStep:
    def dict(self):
        return asdict(self)

    def to_messages(self, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError


@dataclass
class ActionStep(MemoryStep):
    model_input_messages: List[Message] | None = None
    tool_calls: List[ToolCall] | None = None
    start_time: float | None = None
    end_time: float | None = None
    step_number: int | None = None
    error: AgentError | None = None
    duration: float | None = None
    model_output_message: ChatMessage = None
    model_output: str | None = None
    observations: str | None = None
    observations_images: List[str] | None = None
    action_output: Any = None
    include_in_validation: bool = False  # Flag for validation inclusion

    def dict(self):
        # We overwrite the method to parse the tool_calls and action_output manually
        return {
            "model_input_messages": self.model_input_messages,
            "tool_calls": [tc.dict() for tc in self.tool_calls] if self.tool_calls else [],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "step": self.step_number,
            "error": self.error.dict() if self.error else None,
            "duration": self.duration,
            "model_output_message": self.model_output_message,
            "model_output": self.model_output,
            "observations": self.observations,
            "action_output": make_json_serializable(self.action_output),
            "include_in_validation": self.include_in_validation,
        }

    def to_messages(self, summary_mode: bool = False, show_model_input_messages: bool = False) -> List[Message]:
        messages = []
        if self.model_input_messages is not None and show_model_input_messages:
            messages.append(Message(role=MessageRole.SYSTEM, content=self.model_input_messages))
        if self.model_output is not None and not summary_mode:
            messages.append(
                Message(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.model_output.strip()}])
            )

        if self.tool_calls is not None:
            messages.append(
                Message(
                    role=MessageRole.TOOL_CALL,
                    content=[
                        {
                            "type": "text",
                            "text": "Calling tools:\n" + str([tc.dict() for tc in self.tool_calls]),
                        }
                    ],
                )
            )

        if self.observations is not None:
            messages.append(
                Message(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[
                        {
                            "type": "text",
                            "text": (f"Call id: {self.tool_calls[0].id}\n" if self.tool_calls else "")
                            + f"Observation:\n{self.observations}",
                        }
                    ],
                )
            )
        if self.error is not None:
            # For max steps error, don't suggest retry
            if isinstance(self.error, AgentMaxStepsError):
                error_message = (
                    "Error:\n"
                    + str(self.error)
                )
            else:
                error_message = (
                    "Error:\n"
                    + str(self.error)
                    + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
                )
            message_content = f"Call id: {self.tool_calls[0].id}\n" if self.tool_calls else ""
            message_content += error_message
            messages.append(
                Message(role=MessageRole.TOOL_RESPONSE, content=[{"type": "text", "text": message_content}])
            )

        if self.observations_images:
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=[{"type": "text", "text": "Here are the observed images:"}]
                    + [
                        {
                            "type": "image",
                            "image": image,
                        }
                        for image in self.observations_images
                    ],
                )
            )
        
        # Add rethink information if available
        if hasattr(self, 'rethink_info') and self.rethink_info:
            rethink_info = self.rethink_info
            
            # Handle two-stage rethink: reasoning validation stage
            if 'reasoning_rethink_content' in rethink_info:
                messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=[{
                            "type": "text",
                            "text": f"[REASONING VALIDATION]:\n{rethink_info['reasoning_rethink_content']}"
                        }]
                    )
                )
                
                if 'reasoning_rethink_final_answer' in rethink_info:
                    messages.append(
                        Message(
                            role=MessageRole.TOOL_RESPONSE,
                            content=[{
                                "type": "text",
                                "text": f"[REASONING RESULT]:\n{rethink_info['reasoning_rethink_final_answer']}"
                            }]
                        )
                    )
            
            # Handle two-stage rethink: format checking stage
            if 'format_rethink_content' in rethink_info:
                messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=[{
                            "type": "text",
                            "text": f"[FORMAT CHECKING]:\n{rethink_info['format_rethink_content']}"
                        }]
                    )
                )
                
                if 'format_rethink_final_answer' in rethink_info:
                    messages.append(
                        Message(
                            role=MessageRole.TOOL_RESPONSE,
                            content=[{
                                "type": "text",
                                "text": f"[FORMAT RESULT]:\n{rethink_info['format_rethink_final_answer']}"
                            }]
                        )
                    )
            
            # Legacy single-stage rethink support (for backward compatibility)
            elif 'rethink_content' in rethink_info:
                messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=[{
                            "type": "text",
                            "text": f"[RETHINK RESPONSE]:\n{rethink_info['rethink_content']}"
                        }]
                    )
                )
                
                # Add rethink execution if available
                if 'rethink_code' in rethink_info:
                    messages.append(
                        Message(
                            role=MessageRole.TOOL_CALL,
                            content=[{
                                "type": "text",
                                "text": f"[RETHINK EXECUTION]:\nExecuting rethink code:\n{rethink_info['rethink_code']}"
                            }]
                        )
                    )
                    
                    # Add rethink execution result
                    if 'rethink_final_answer' in rethink_info:
                        result_text = f"[RETHINK RESULT]:\nFinal answer: {rethink_info['rethink_final_answer']}"
                        if 'rethink_execution_logs' in rethink_info and rethink_info['rethink_execution_logs']:
                            result_text += f"\nExecution logs:\n{rethink_info['rethink_execution_logs']}"
                        
                        messages.append(
                            Message(
                                role=MessageRole.TOOL_RESPONSE,
                                content=[{
                                    "type": "text",
                                    "text": result_text
                                }]
                            )
                        )
        
        return messages


@dataclass
class PlanningStep(MemoryStep):
    model_input_messages: List[Message]
    model_output_message_facts: ChatMessage = None
    facts: str = None
    model_output_message_plan: ChatMessage = None
    plan: str = None
    model_output_message_decomposition: ChatMessage = None
    decomposed_tasks: str = None

    def to_messages(self, summary_mode: bool, **kwargs) -> List[Message]:
        messages = []
        
        # Handle new decomposition approach
        if self.decomposed_tasks is not None:
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT, content=[{"type": "text", "text": f"[TASK DECOMPOSITION]:\n{self.decomposed_tasks.strip()}"}]
                )
            )
        
        # Handle legacy facts and plan approach for backward compatibility
        if self.facts is not None:
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT, content=[{"type": "text", "text": f"[FACTS LIST]:\n{self.facts.strip()}"}]
                )
            )

        if self.plan is not None and not summary_mode:  # This step is not shown to a model writing a plan to avoid influencing the new plan
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT, content=[{"type": "text", "text": f"[PLAN]:\n{self.plan.strip()}"}]
                )
            )
        return messages


@dataclass
class TaskStep(MemoryStep):
    task: str
    task_images: List[str] | None = None

    def to_messages(self, summary_mode: bool = False, **kwargs) -> List[Message]:
        content = [{"type": "text", "text": f"New task:\n{self.task}"}]
        if self.task_images:
            for image in self.task_images:
                content.append({"type": "image", "image": image})

        return [Message(role=MessageRole.USER, content=content)]


@dataclass
class SystemPromptStep(MemoryStep):
    system_prompt: str

    def to_messages(self, summary_mode: bool = False, **kwargs) -> List[Message]:
        if summary_mode:
            return []
        return [Message(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}])]


class AgentMemory:
    def __init__(self, system_prompt: str):
        self.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        self.steps: List[Union[TaskStep, ActionStep, PlanningStep]] = []

    def reset(self):
        self.steps = []

    def get_succinct_steps(self) -> list[dict]:
        return [
            {key: value for key, value in step.dict().items() if key != "model_input_messages"} for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        return [step.dict() for step in self.steps]

    def replay(self, logger: AgentLogger, detailed: bool = False):
        """Prints a pretty replay of the agent's steps.

        Args:
            logger (AgentLogger): The logger to print replay logs to.
            detailed (bool, optional): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        logger.console.log("Replaying the agent's steps:")
        for step in self.steps:
            if isinstance(step, SystemPromptStep) and detailed:
                logger.log_markdown(title="System prompt", content=step.system_prompt, level=LogLevel.ERROR)
            elif isinstance(step, TaskStep):
                logger.log_task(step.task, "", level=LogLevel.ERROR)
            elif isinstance(step, ActionStep):
                logger.log_rule(f"Step {step.step_number}", level=LogLevel.ERROR)
                if detailed:
                    logger.log_messages(step.model_input_messages)
                logger.log_markdown(title="Agent output:", content=step.model_output, level=LogLevel.ERROR)
            elif isinstance(step, PlanningStep):
                logger.log_rule("Planning step", level=LogLevel.ERROR)
                if detailed:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                
                # Handle both new decomposition and legacy approaches
                content = ""
                if step.decomposed_tasks:
                    content += step.decomposed_tasks
                if step.facts:
                    content += "\n" + step.facts if content else step.facts
                if step.plan:
                    content += "\n" + step.plan if content else step.plan
                    
                logger.log_markdown(title="Agent output:", content=content, level=LogLevel.ERROR)


__all__ = ["AgentMemory"]
