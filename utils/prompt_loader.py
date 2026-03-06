from utils.config_handler import prompt_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


def load_system_prompt() -> str:
    try:
        system_prompt_path = get_abs_path(prompt_config["main_prompt_path"])

    except KeyError as e:
        logger.error(f"Prompt name 'main_prompt_path' not found in prompt_config: {e}")
        raise e
    
    try:
        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        return system_prompt
    except Exception as e:
        logger.error(f"Error loading prompt from {system_prompt_path}: {e}")
        raise e
    

def load_rag_prompt() -> str:
    try:
        rag_prompt_path = get_abs_path(prompt_config["rag_summary_prompt_path"])

    except KeyError as e:
        logger.error(f"Prompt name 'rag_summary_prompt_path' not found in prompt_config: {e}")
        raise e
    
    try:
        with open(rag_prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        return system_prompt
    except Exception as e:
        logger.error(f"Error loading prompt from {rag_prompt_path}: {e}")
        raise e
    

def load_report_prompt() -> str:
    try:
        report_prompt_path = get_abs_path(prompt_config["report_prompt_path"])

    except KeyError as e:
        logger.error(f"Prompt name 'report_prompt_path' not found in prompt_config: {e}")
        raise e
    
    try:
        with open(report_prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        return system_prompt
    except Exception as e:
        logger.error(f"Error loading prompt from {report_prompt_path}: {e}")
        raise e
    

if __name__ == "__main__":
    print(load_system_prompt())
    print(load_rag_prompt())
    print(load_report_prompt())
