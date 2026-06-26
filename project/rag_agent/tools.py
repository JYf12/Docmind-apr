from typing import List
from langchain_core.tools import tool
from db.parent_store_manager import ParentStoreManager

from flashrank import Ranker, RerankRequest   # 新增              flashrank:集成了cross-encoder等重排序算法的工具库
from langchain_core.documents import Document  # 新增

class ToolFactory:
    
    def __init__(self, collection):
        self.collection = collection
        self.parent_store_manager = ParentStoreManager()

        # ================== 新增：初始化 Reranker ==================
        self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./cache")  # 轻量版
        # 更强模型：
        # self.reranker = Ranker(model_name="BAAI/bge-reranker-base", cache_dir="./cache")
        # ============================================================
    
    def _search_child_chunks(self, query: str, limit: int) -> str:
        """Search for the top K most relevant child chunks.

        Args:
            query: Search query string
            limit: Maximum number of results to return
        """
        try:
            results = self.collection.similarity_search(query, k=limit*2, score_threshold=0.45)
            if not results:
                return "NO_RELEVANT_CHUNKS"

            # Step 2: 转换为 Document 格式供 reranker 使用
            docs = [
                Document(
                    page_content=doc.page_content,
                    metadata=doc.metadata
                ) for doc in results
            ]

            # Step 3: 执行 Rerank（关键）
            reranked_docs = self._rerank_documents(query, docs, top_n=5)

            return "\n\n".join([
                f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                f"File Name: {doc.metadata.get('source', '')}\n"
                f"Content: {doc.page_content.strip()}"
                for doc in reranked_docs
            ])            

        except Exception as e:
            return f"RETRIEVAL_ERROR: {str(e)}"

        # ================== 新增 Rerank 方法 ==================
    def _rerank_documents(self, query: str, docs: List[Document], top_n: int = 5):
        """使用 FlashRank 对文档进行重排序"""
        if not docs:
            return []

        passages = [
            {"id": i, "text": doc.page_content, "metadata": doc.metadata}
            for i, doc in enumerate(docs)
        ]

        request = RerankRequest(query=query, passages=passages)
        results = self.reranker.rerank(request)

        # 取 top_n
        reranked = []
        for res in results[:top_n]:
            original_doc = docs[res['id']]
            reranked.append(original_doc)

        return reranked
    
    def _retrieve_many_parent_chunks(self, parent_ids: List[str]) -> str:           # 多个父块内容召回  根据parent_id集合召回所有父块内容
        """Retrieve full parent chunks by their IDs.
    
        Args:
            parent_ids: List of parent chunk IDs to retrieve
        """
        try:
            ids = [parent_ids] if isinstance(parent_ids, str) else list(parent_ids)
            raw_parents = self.parent_store_manager.load_content_many(ids)
            if not raw_parents:
                return "NO_PARENT_DOCUMENTS"

            return "\n\n".join([
                f"Parent ID: {doc.get('parent_id', 'n/a')}\n"
                f"File Name: {doc.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {doc.get('content', '').strip()}"
                for doc in raw_parents
            ])            

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
    
    def _retrieve_parent_chunks(self, parent_id: str) -> str:                       # 单个父块内容召回 根据单个parent_id召回单个父块内容
        """Retrieve full parent chunks by their IDs.
    
        Args:
            parent_id: Parent chunk ID to retrieve
        """
        try:
            parent = self.parent_store_manager.load_content(parent_id)
            if not parent:
                return "NO_PARENT_DOCUMENT"

            return (
                f"Parent ID: {parent.get('parent_id', 'n/a')}\n"
                f"File Name: {parent.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {parent.get('content', '').strip()}"
            )          

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
    
    def create_tools(self) -> List:
        """Create and return the list of tools."""
        search_tool = tool("search_child_chunks")(self._search_child_chunks)                # 搜索工具---子块相似度检索
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)        # 召回工具---单个父块内容召回
        
        return [search_tool, retrieve_tool]         # 返回工具列表[搜索工具+召回工具]