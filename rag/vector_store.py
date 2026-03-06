import os
import sys

if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from langchain_chroma import Chroma
from utils.config_handler import chroma_config
from model.factory import get_embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from langchain_core.documents import Document


class VectorStoreService:
    def __init__(self):
        persist_directory = get_abs_path(chroma_config["persist_directory"])
        os.makedirs(persist_directory, exist_ok=True)
        self.vector_store = Chroma(
            collection_name=chroma_config["collection_name"],
            persist_directory=persist_directory,
            embedding_function=get_embedding_model()
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"], 
            chunk_overlap=chroma_config["chunk_overlap"],
            separators=chroma_config["separators"]
            )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_config["k"]})
    
    def load_document(self) -> int:
        def check_md5_hex(md5_str_hex: str):
            if not os.path.exists(get_abs_path(chroma_config["md5_hex_store"])):
                open(get_abs_path(chroma_config["md5_hex_store"]), 'w').close()
                return False
            with open(get_abs_path(chroma_config["md5_hex_store"]), 'r') as f:
                for line in f:
                    if line.strip() == md5_str_hex:
                        return True
            return False
        

        def save_md5_hex(md5_str_hex: str):
            with open(get_abs_path(chroma_config["md5_hex_store"]), 'a') as f:
                f.write(md5_str_hex + '\n') 

        def get_file_documents(file_path: str):
            if file_path.lower().endswith(".pdf"):
                return pdf_loader(file_path)
            elif file_path.lower().endswith(".txt"):
                return txt_loader(file_path)
            else:
                logger.warning(f"Unsupported file type for {file_path}. Skipping.")
                return []
            
        allowed_files_path = listdir_with_allowed_type(
            get_abs_path(chroma_config["data_path"]), 
            tuple(
                chroma_config.get("allow_knowledge_file_type")
                or []
            )
            )
        processed_count = 0
        for file_path in allowed_files_path:
            file_md5_hex = get_file_md5_hex(file_path)
            if check_md5_hex(file_md5_hex):
                logger.info(f"{file_path} has already been processed. Skipping.")
                continue
            try:
                documents = get_file_documents(file_path)
                if not documents:
                    logger.warning(f"[Load KB]No documents found in {file_path}. Skipping.")
                    continue
                split_document: list[Document] =self.spliter.split_documents(documents)
                if not split_document:
                    logger.warning(f"[Load KB]Failed to split documents from {file_path}. Skipping.")
                    continue
                self.vector_store.add_documents(split_document)
                save_md5_hex(file_md5_hex)
                processed_count += 1
                logger.info(f"[Load KB]Successfully processed {file_path} and added to vector store.")
            except Exception as e:
                logger.error(f"[Load KB]Error processing {file_path}: {e}", exc_info=True)  # log exact error details
                continue
        return processed_count


if __name__ == "__main__":
    vector_store = VectorStoreService()
    loaded_count = vector_store.load_document()
    if loaded_count <= 0:
        logger.warning("No documents were added to the vector store. Skipping retrieval.")
    else:
        retriever = vector_store.get_retriever()
        try:
            res = retriever.invoke("迷路")
            for doc in res:
                print(doc.page_content)
        except Exception as e:
            logger.error(f"Retriever query failed: {e}", exc_info=True)
