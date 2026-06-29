import os
# os.environ['HF_HUB_OFFLINE'] = '1'
import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

class VectorDbManager:
    __client: QdrantClient
    __dense_embeddings: HuggingFaceEmbeddings           # Dense嵌入模型
    __sparse_embeddings: FastEmbedSparse                # 稀疏嵌入模型
    def __init__(self):
        self.__client = QdrantClient(path=config.QDRANT_DB_PATH)
        self.__dense_embeddings = HuggingFaceEmbeddings(model_name=config.DENSE_MODEL)          # 创建Dense密集嵌入模型
        self.__sparse_embeddings = FastEmbedSparse(model_name=config.SPARSE_MODEL)              # 创建Sparse稀疏嵌入模型

    def create_collection(self, collection_name):
        """创建向量存储集合"""
        if not self.__client.collection_exists(collection_name):
            print(f"Creating collection: {collection_name}...")
            self.__client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(size=len(self.__dense_embeddings.embed_query("test")), distance=qmodels.Distance.COSINE),       # 密集向量参数
                sparse_vectors_config={config.SPARSE_VECTOR_NAME: qmodels.SparseVectorParams()},                                                    # 稀疏参数
            )
            print(f"✓ Collection created: {collection_name}")
        else:
            print(f"✓ Collection already exists: {collection_name}")

    def delete_collection(self, collection_name):
        try:
            if self.__client.collection_exists(collection_name):
                print(f"Removing existing Qdrant collection: {collection_name}")
                self.__client.delete_collection(collection_name)
        except Exception as e:
            print(f"Warning: could not delete collection {collection_name}: {e}")

    def get_collection(self, collection_name) -> QdrantVectorStore:
        try:
            return QdrantVectorStore(
                    client=self.__client,
                    collection_name=collection_name,
                    embedding=self.__dense_embeddings,
                    sparse_embedding=self.__sparse_embeddings,
                    retrieval_mode=RetrievalMode.HYBRID,        # 混合检索模式
                    sparse_vector_name=config.SPARSE_VECTOR_NAME
                )
        except Exception as e:
            print(f"Unable to get collection {collection_name}: {e}")