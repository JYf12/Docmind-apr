from pathlib import Path
import shutil
import config
from utils import pdfs_to_markdowns, clear_directory_contents

class DocumentManager:

    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.markdown_dir = Path(config.MARKDOWN_DIR)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        
    def add_documents(self, document_paths, progress_callback=None):
        """
        1.分层分块处理---返回父块，子块
        2.子块存入向量数据库，父块本地化json存储

        input:上传文件路径
        output:添加文件数，跳过文件数
        """
        if not document_paths:
            return 0, 0
            
        document_paths = [document_paths] if isinstance(document_paths, str) else document_paths
        document_paths = [p for p in document_paths if p and Path(p).suffix.lower() in [".pdf", ".md"]]     # 过滤不是Markdown或PDF的文件
        
        if not document_paths:
            return 0, 0
            
        added = 0
        skipped = 0
            
        for i, doc_path in enumerate(document_paths):
            if progress_callback:
                progress_callback((i + 1) / len(document_paths), f"Processing {Path(doc_path).name}")       # 展示进度条
                
            doc_name = Path(doc_path).stem
            md_path = self.markdown_dir / f"{doc_name}.md"
            
            if md_path.exists():        # 如果文件已存在，则跳过
                skipped += 1
                continue
                
            try:            
                if Path(doc_path).suffix.lower() == ".md":      # 如果是Markdown文件，则直接复制以进行分块
                    shutil.copy(doc_path, md_path)
                else:
                    pdfs_to_markdowns(str(doc_path), overwrite=False)          # 不是Markdown文件则将PDF转换为Markdown
                parent_chunks, child_chunks = self.rag_system.chunker.create_chunks_single(md_path)         # 分层分块实现
                
                if not child_chunks:
                    skipped += 1
                    continue
                
                collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)      # 获取向量数据库
                collection.add_documents(child_chunks)                                                      # 向量数据库中添加子块
                self.rag_system.parent_store.save_many(parent_chunks)                                       # 父块以json格式保存到本地
                
                added += 1
                
            except Exception as e:
                print(f"Error processing {doc_path}: {e}")
                skipped += 1
            
        return added, skipped
    
    def get_markdown_files(self):
        """ 遍历markdown文件目录并以.pdf的后缀形式返回所有.md文件 """
        if not self.markdown_dir.exists():
            return []
        return sorted([p.name.replace(".md", ".pdf") for p in self.markdown_dir.glob("*.md")])
    
    def clear_all(self):
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        clear_directory_contents(self.markdown_dir)             # 清空markdown文件目录
        
        self.rag_system.parent_store.clear_store()              # 删除父块存储json文件
        self.rag_system.vector_db.delete_collection(self.rag_system.collection_name)       # 删除Qdrant向量存储集合
        self.rag_system.vector_db.create_collection(self.rag_system.collection_name)       # 重新创建向量数据库空集合