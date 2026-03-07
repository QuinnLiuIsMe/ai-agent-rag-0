# AI Agent RAG Backend

This project is a Python backend for:
- RAG-based Q&A on robot-vacuum knowledge files (`txt`/`pdf`)
- A ReAct-style agent that can call tools (RAG summary, weather, user/report data)
- Dynamic prompt switching for report scenarios

## 1. Project Structure

```text
backend/
├── agent/
│   ├── react_agent.py              # ReAct agent entry
│   └── tools/
│       ├── agent_tools.py          # Tool definitions
│       └── middleware.py           # Agent middleware (logging, prompt switch)
├── rag/
│   ├── vector_store.py             # Build/load Chroma vector DB
│   └── rag_service.py              # RAG summarize service
├── model/
│   └── factory.py                  # Chat/embedding model factories
├── utils/                          # Config, logger, file/path helpers
├── config/                         # rag/chroma/prompt/agent YAML configs
├── prompts/                        # Main/RAG/report prompt templates
├── data/                           # Knowledge files + external report CSV
├── chroma_db/                      # Persisted vector DB (generated)
├── logs/                           # Runtime logs (generated)
└── requirement.txt
```

## 2. How It Works

### 2.1 Vector Store (`rag/vector_store.py`)
- Loads files from `config/chroma.yml -> data_path`
- Supports `.txt` and `.pdf`
- Splits documents with `RecursiveCharacterTextSplitter`
- Embeds and stores chunks in Chroma
- Uses `md5.txt` to skip already-processed files

### 2.2 RAG Summarization (`rag/rag_service.py`)
- Retrieves top-`k` chunks from Chroma
- Builds context as `Reference Document [n]: ...`
- Injects `{input}` + `{context}` into `prompts/rag_summarize.txt`
- Calls chat model to produce final Chinese summary

### 2.3 Agent (`agent/react_agent.py`)
- Creates LangChain agent with:
  - tools from `agent/tools/agent_tools.py`
  - middleware from `agent/tools/middleware.py`
  - system prompt from `prompts/main_prompt.txt`
- Streams model/tool responses
- If DashScope is unreachable, returns a graceful fallback message

### 2.4 Middleware (`agent/tools/middleware.py`)
- `monitor_tool`: logs tool name/args and toggles `context["report"]`
- `log_before_model`: logs message count before model call
- `reqport_prompt_switch`: switches prompt to report template when report mode is enabled

### 2.5 Model Factory (`model/factory.py`)
- Chat model: `ChatTongyi(model_name from config/rag.yml)`
- Embedding model:
  - default `DashScopeEmbeddings`
  - optional fallback `FakeEmbeddings` when network/API is unavailable

## 3. Setup

## Requirements
- Python 3.12 recommended
- DashScope API key for online model calls

Install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

Set API key:

```bash
export DASHSCOPE_API_KEY="your_key"
```

## 4. Run

### 4.1 Build/refresh vector DB

```bash
python rag/vector_store.py
```

### 4.2 Run RAG summarize demo

```bash
python rag/rag_service.py
```

### 4.3 Run ReAct agent demo

```bash
python agent/react_agent.py
```

## 5. Config Quick Reference

### `config/rag.yml`
- `model_name`: chat model name (e.g. `qwen3-max`)
- `embedding_model_name`: embedding model (e.g. `text-embedding-v4`)
- optional `embedding_provider: fake` for offline embedding tests

### `config/chroma.yml`
- `collection_name`, `persist_directory`
- `k` retriever top-k
- `data_path`, `allow_knowledge_file_type`
- `chunk_size`, `chunk_overlap`, `separators`

### `config/prompt.yml`
- paths for main/rag/report prompts

### `config/agent.yml`
- `external_data_path` for report CSV

## 6. Data Flow Summary

1. Ingest local docs to Chroma (`vector_store.py`)
2. User query enters `rag_service.py` or `react_agent.py`
3. Retriever gets related chunks from Chroma
4. Prompt template combines query + retrieved context
5. LLM generates answer/report

## 7. Known Limitations

- Project depends on DashScope network access for chat generation.
- If DNS/network to `dashscope.aliyuncs.com` fails:
  - embeddings may fall back to `FakeEmbeddings`
  - chat generation will return fallback error message
