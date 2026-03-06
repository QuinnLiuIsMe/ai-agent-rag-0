import os
import hashlib
from utils.logger_handler import logger
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# md5 deduplication
def get_file_md5_hex(file_path: str):
    if not os.path.exists(file_path):
        logger.error(f"{file_path} does not exist.[md5]")
        raise ValueError(f"{file_path} is not a valid file.")
    if not os.path.isfile(file_path):
        logger.error(f"{file_path} is not a file.[md5]")
        raise ValueError(f"{file_path} is not a valid file.")
    
    md5_obj = hashlib.md5()
    chunk_size = 4096  # 4KB, avoid loading the entire file into memory
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
        return md5_obj.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating md5 for {file_path}: {e}")
        raise e 



def listdir_with_allowed_type(dir_path: str, allowed_types: tuple[str]):
    files = []
    if not os.path.isdir(dir_path):
        logger.error(f"{dir_path} does not exist.[listdir]")
        raise ValueError(f"{dir_path} is not a valid directory.")
    
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            if filename.lower().endswith(allowed_types):
                files.append(os.path.join(dir_path, filename))
    return files

def pdf_loader(file_path: str, password: str = None):
    return PyPDFLoader(file_path, password=password).load()


def txt_loader(file_path: str):
    return TextLoader(file_path).load()
