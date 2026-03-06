import os
import sys
from typing import Callable

if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from langchain.agents.middleware import (
    AgentState,
    ModelRequest,
    before_model,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompt, load_system_prompt


@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest, 
        handler: Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> ToolMessage | Command:

    logger.info(f"[tool monitor] 执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor] 传入参数：{request.tool_call['args']}")
    try:
        res = handler(request)
        logger.info(f"[tool monitor] 调用成功{request.tool_call['name']}")
        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True
        return res
    except Exception as e:
        logger.info(f"[tool monitor] 调用失败：{request.tool_call['name']}")
        raise e
    

@before_model
def log_before_model(
    state: AgentState,
    runtime: Runtime,
):
    logger.info(f"[log before model] 即将调用模型，带有信息条数：{len(state['messages'])}")
    logger.debug(f"[log before model] {type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")


@dynamic_prompt
def reqport_prompt_switch(request: ModelRequest):
    is_report = request.runtime.context.get("report", False)
    if is_report:
        return load_report_prompt()
    return load_system_prompt()
