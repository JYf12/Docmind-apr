import uuid
from langchain_ollama import ChatOllama
import config
from db.vector_db_manager import VectorDbManager
from db.parent_store_manager import ParentStoreManager
from document_chunker import DocumentChuncker
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph
from core.observability import Observability
from core.image_manager import ImageSummaryGenerator, ImageStoreManager

class RAGSystem:

    def __init__(self, collection_name=config.CHILD_COLLECTION):
        self.collection_name = collection_name
        self.vector_db = VectorDbManager()
        self.parent_store = ParentStoreManager()
        self.chunker = DocumentChuncker()
        self.observability = Observability()
        self.image_summary_generator = ImageSummaryGenerator() if config.IMAGE_SUMMARY_ENABLED else None
        self.image_store_manager = ImageStoreManager()
        self.agent_graph = None
        self.multimodal_llm = None      # lazy init in initialize()
        self.thread_id = str(uuid.uuid4())
        self.recursion_limit = config.GRAPH_RECURSION_LIMIT

    def _init_multimodal_llm(self):
        """Lazy-init the multimodal LLM for final answer synthesis.
        Returns None when no multimodal provider is configured → pure-text fallback."""
        if self.multimodal_llm is not None:
            return self.multimodal_llm

        provider = config.MULTIMODAL_LLM_PROVIDER or config.ACTIVE_LLM_CONFIG
        model = config.MULTIMODAL_LLM_MODEL

        try:
            if provider == "openai":
                from langchain_openai import ChatOpenAI
                self.multimodal_llm = ChatOpenAI(
                    model=model,
                    temperature=config.LLM_CONFIGS.get("openai", {}).get("temperature", 0),
                    api_key=config.LLM_API_KEY,
                    base_url=config.LLM_API_URL,
                )
            elif provider == "ollama":
                print("Using Ollama as multimodal LLM")
                from langchain_openai import ChatOpenAI
                active_cfg = config.LLM_CONFIGS.get("ollama", {})
                self.multimodal_llm = ChatOpenAI(
                    model=model,
                    temperature=0,
                    api_key="ollama",
                    base_url=active_cfg.get("url", "http://localhost:11434") + "/v1",
                )
            elif provider == "qwen":
                print("Using Qwen as multimodal LLM")
                from langchain_openai import ChatOpenAI
                active_cfg = config.LLM_CONFIGS.get("qwen", {})
                self.multimodal_llm = ChatOpenAI(
                    model=model,
                    temperature=0,
                    api_key=active_cfg.get("api_key", ""),
                    base_url=active_cfg.get("base_url", ""),
                )
            # elif provider == "anthropic":
            #     from langchain_anthropic import ChatAnthropic
            #     self.multimodal_llm = ChatAnthropic(model=model, temperature=0)
            # elif provider == "google":
            #     from langchain_google_genai import ChatGoogleGenerativeAI
            #     self.multimodal_llm = ChatGoogleGenerativeAI(model=model, temperature=0)
            else:
                print(f"⚠ Unsupported MULTIMODAL_LLM_PROVIDER: {provider} — multimodal synthesis disabled")
                self.multimodal_llm = None
        except Exception as e:
            print(f"⚠ Failed to init multimodal LLM: {e} — multimodal synthesis disabled")
            self.multimodal_llm = None

        return self.multimodal_llm

    def initialize(self):
        self.vector_db.create_collection(self.collection_name)          # 1.创建向量数据库
        collection = self.vector_db.get_collection(self.collection_name)

        # llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
        # Load active configuration
        active_config = config.LLM_CONFIGS[config.ACTIVE_LLM_CONFIG]
        model = active_config["model"]
        temperature = active_config["temperature"]

        if config.ACTIVE_LLM_CONFIG == "ollama":
            # from langchain_ollama import ChatOllama
            # llm = ChatOllama(model=model, temperature=temperature, base_url=active_config["url"])

            # 使用 LangChain 的 ChatOpenAI 包装，自动集成 Langfuse 追踪
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key="ollama",
                base_url=active_config.get("url", "http://localhost:11434") + "/v1",
            )


        elif config.ACTIVE_LLM_CONFIG == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=config.LLM_API_KEY,
                base_url=config.LLM_API_URL,
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {config.ACTIVE_LLM_CONFIG}")

        # Init multimodal LLM (may be None — triggers pure-text fallback)
        multimodal_llm = self._init_multimodal_llm()

        tools = ToolFactory(collection, image_store_manager=self.image_store_manager).create_tools()
        self.agent_graph = create_agent_graph(llm, tools, multimodal_llm=multimodal_llm)

    def get_config(self):
        cfg = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": self.recursion_limit}
        handler = self.observability.get_handler()
        if handler:
            cfg["callbacks"] = [handler]
        return cfg

    def reset_thread(self):
        try:
            self.agent_graph.checkpointer.delete_thread(self.thread_id)
        except Exception as e:
            print(f"Warning: Could not delete thread {self.thread_id}: {e}")
        self.thread_id = str(uuid.uuid4())