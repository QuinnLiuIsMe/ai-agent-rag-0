import yaml
from utils.path_tool import get_abs_path


def _load_yaml_config(config_path: str, encoding: str = "utf-8") -> dict:
    with open(get_abs_path(config_path), "r", encoding=encoding) as file:
        return yaml.safe_load(file) or {}


def load_rag_config(
    config_path: str = "config/rag.yml",
    encoding: str = "utf-8",
) -> dict:
    config = _load_yaml_config(config_path, encoding)
    if "model_name" not in config:
        if "chat_model_name" in config:
            config["model_name"] = config["chat_model_name"]
        elif "chat_moldel_name" in config:
            config["model_name"] = config["chat_moldel_name"]
    return config

def load_chroma_config(
    config_path: str = "config/chroma.yml",
    encoding: str = "utf-8",
) -> dict:
    config = _load_yaml_config(config_path, encoding)
    if "allow_knowledge_file_type" not in config:
        config["allow_knowledge_file_type"] = config["allow_knnowledge_file_type"]
    return config

def load_prompt_config(
    config_path: str = "config/prompt.yml",
    encoding: str = "utf-8",
) -> dict:
    return _load_yaml_config(config_path, encoding)

def load_agent_config(
    config_path: str = "config/agent.yml",
    encoding: str = "utf-8",
) -> dict:
    return _load_yaml_config(config_path, encoding)


agent_config = load_agent_config()
rag_config = load_rag_config()
chroma_config = load_chroma_config()
prompt_config = load_prompt_config()
