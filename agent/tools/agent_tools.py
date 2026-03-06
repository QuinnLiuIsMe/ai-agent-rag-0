import os
import sys

if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from langchain_core.tools import tool
import random 
from utils.config_handler import agent_config
from collections import defaultdict
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from rag.rag_service import RagSummarizeService


rag = RagSummarizeService()
user_ids = [str(i) for i in range(1001, 1011)]
month_arr = ["2025-" + ("0" + str(i))[-2:] for i in range(1, 13)] 

external_data = defaultdict(dict)


@tool(description="Summarize knowledge-base context for a user query.")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="Return mock weather for a city.")
def get_weather(city: str) -> str:
    return f"the city {city} wweather is Sunny."


@tool(description="Return a random mock user location.")
def get_user_location() -> str:
    return random.choice(['Beijing', 'Shanghai', 'Hongkong'])


@tool
def get_user_id() -> str:
    """Return a random mock user id."""
    return random.choice(user_ids)


@tool
def get_cur_month() -> str:
    """Return a random month string."""
    return random.choice(month_arr)


@tool
def generate_external_data() -> None:
    """Internal loader for external user data."""
    if not external_data:
        external_data_path = agent_config.get("external_data_path")
        if not external_data_path:
            logger.warning("`external_data_path` missing in config/agent.yml")
            return

        external_data_path = get_abs_path(external_data_path)
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"No external file: {external_data_path}")
        
        with open(external_data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")
                user_id: str = arr[0].replace('"', '')
                feature: str = arr[1].replace('"', '')
                efficiency: str = arr[2].replace('"', '')
                consumables: str = arr[3].replace('"', '')
                comparison: str = arr[4].replace('"', '')
                time: str = arr[5].replace('"', '')

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool
def fetch_external_data(user_id: str, month: str):
    """Fetch one user's external data by month from cache."""
    generate_external_data()
    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data] cannot find data of user {user_id} in month {month}")


@tool
def fill_context_for_report():
    """Mark runtime context so report prompt can be selected."""
    return "已调用fill_context_for_report"
