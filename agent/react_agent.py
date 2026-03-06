import os
import sys

if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from langchain.agents import create_agent
from model.factory import get_chat_model
from utils.prompt_loader import load_system_prompt
from utils.logger_handler import logger
from agent.tools.agent_tools import rag_summarize, get_weather, get_user_location, get_user_id, get_cur_month, fetch_external_data, fill_context_for_report
from agent.tools.middleware import monitor_tool, log_before_model, reqport_prompt_switch


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=get_chat_model(),
            system_prompt=load_system_prompt(),
            tools=[rag_summarize, get_weather, get_user_location,
                   get_user_id, get_cur_month, fetch_external_data, 
                   fill_context_for_report],
            middleware=[monitor_tool, log_before_model, reqport_prompt_switch]
        )

    def execute_stream(self, query: str):
        input_dict = {
            "messages" : [
                {"role": "user", "content": query}
            ]
        }
        try:
            for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):  # context is for prompt switching
                latest_message = chunk["messages"][-1]
                if latest_message.content:
                    yield latest_message.content.strip() + "\n"
        except Exception as e:
            logger.error(f"ReactAgent stream failed: {e}", exc_info=True)
            yield "当前无法连接模型服务（DashScope），请检查网络/DNS与 DASHSCOPE_API_KEY 后重试。\n"


if __name__ == "__main__":
    ra = ReactAgent()
    for chunk in ra.execute_stream("扫地机器人在我所在的地区的气温下如何保养"):
        print(chunk, end="", flush=True)
