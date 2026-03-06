try:
    from rag.vector_store import VectorStoreService
except ModuleNotFoundError:
    from vector_store import VectorStoreService
from utils.logger_handler import logger
from model.factory import get_chat_model
from langchain_core.prompts import PromptTemplate
from utils.prompt_loader import load_rag_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


class RagSummarizeService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriver = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = get_chat_model()
        self.chain = self._init_chian()

    def _init_chian(self):
        chain = self.prompt_template | self._print_prompt_template | self.model | StrOutputParser()
        return chain
    
    def retriver_docs(self, query: str) -> list[Document]:
        return self.retriver.invoke(query)
    
    def rag_summarize(self, query: str) -> str:
        docs = self.retriver_docs(query)
        if not docs:
            logger.warning("No relevant documents found for the query.")
            return "No relevant information found."
        counter = 0
        context = ""
        for doc in docs:
            counter += 1
            context += f"Reference Document [{counter}]: {doc.page_content}\n\n"

        try:
            summary = self.chain.invoke({
                "input": query,
                "context": context
            })
            return summary
        except Exception as e:
            logger.error(f"RAG summarize failed: {e}", exc_info=True)
            return (
                "当前无法连接模型服务（DashScope），无法生成总结。"
                "请检查网络/DNS与 DASHSCOPE_API_KEY 后重试。"
            )
    
    def _print_prompt_template(self, prompt: str):
        print(f"--> Prompt Template:\n{prompt}\n")
        return prompt
    

if __name__ == "__main__":
    rag_service = RagSummarizeService()
    query = "小户型适合哪些扫地机器人?"
    summary = rag_service.rag_summarize(query)
    print(f"Summary:\n{summary}")
