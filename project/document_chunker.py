import os
import glob
import config
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class DocumentChuncker:
    """
    文档分块器的核心实现
    Markdown--->父块(parent chunks),子块(child chunks)
    采用父子分块策略，即保持上下文完整性，又支持细粒度精确检索

    整体策略：
    1.基于Markdown文档结构分块生成初始父块，再通过分块粒度（最大最小阈值）调整父块的长度，解决碎片化问题-----保持上下文
    2.基于已经生成的父块，对每个块采用递归字符分块的策略生成每个小块，用于精细向量检索

    input: pdf文档转换为.md后的文件路径
    output:
        -- all_parent_chunks[(parent_id, chunk)]
        -- all_child_chunks[chunk]
    """
    def __init__(self):
        self.__parent_splitter = MarkdownHeaderTextSplitter(        # 父块分块器---基于文档结构的分块
            headers_to_split_on=config.HEADERS_TO_SPLIT_ON, 
            strip_headers=False
        )
        self.__child_splitter = RecursiveCharacterTextSplitter(     # 子块分块器---递归字符分块
            chunk_size=config.CHILD_CHUNK_SIZE, 
            chunk_overlap=config.CHILD_CHUNK_OVERLAP
        )
        self.__min_parent_size = config.MIN_PARENT_SIZE             # 父块的最小长度
        self.__max_parent_size = config.MAX_PARENT_SIZE             # 父块的最大长度

    def create_chunks(self, path_dir=config.MARKDOWN_DIR):
        all_parent_chunks, all_child_chunks = [], []

        for doc_path_str in sorted(glob.glob(os.path.join(path_dir, "*.md"))):
            doc_path = Path(doc_path_str)
            parent_chunks, child_chunks = self.create_chunks_single(doc_path)
            all_parent_chunks.extend(parent_chunks)
            all_child_chunks.extend(child_chunks)
        
        return all_parent_chunks, all_child_chunks

    def create_chunks_single(self, md_path):
        doc_path = Path(md_path)
        
        with open(doc_path, "r", encoding="utf-8") as f:
            parent_chunks = self.__parent_splitter.split_text(f.read())     # 1.按文档结构分块
        
        merged_parents = self.__merge_small_parents(parent_chunks)          # 2.合并过小的段落
        split_parents = self.__split_large_parents(merged_parents)          # 3.拆分过长的段落(递归字符分块)
        cleaned_parents = self.__clean_small_chunks(split_parents)          # 4.进一步处理剩余的小块

        all_parent_chunks, all_child_chunks = [], []
        self.__create_child_chunks(all_parent_chunks, all_child_chunks, cleaned_parents, doc_path)  # 通过父块生成子块
        return all_parent_chunks, all_child_chunks

    def __merge_small_parents(self, chunks):            # 合并过小的段落: 当一个段落长度小于阈值时，将其与下一个段落合并再继续判断，否则直接成段
        if not chunks:
            return []
        
        merged, current = [], None
        
        for chunk in chunks:
            if current is None:
                current = chunk
            else:
                current.page_content += "\n\n" + chunk.page_content                 # 未满足最小段落长度的段落，将其与下一个段落合并
                for k, v in chunk.metadata.items():                                 # 合并不同段落的元数据
                    if k in current.metadata:                                       # 存在相同元数据，则将其值进行拼接
                        current.metadata[k] = f"{current.metadata[k]} -> {v}"
                    else:
                        current.metadata[k] = v

            if len(current.page_content) >= self.__min_parent_size:
                merged.append(current)
                current = None
        
        if current:                     # 最后一个不满足最小长度的段落，将其强制添加到已经成段的最后一个段落上
            if merged:
                merged[-1].page_content += "\n\n" + current.page_content
                for k, v in current.metadata.items():
                    if k in merged[-1].metadata:
                        merged[-1].metadata[k] = f"{merged[-1].metadata[k]} -> {v}"
                    else:
                        merged[-1].metadata[k] = v
            else:
                merged.append(current)
        
        return merged               # 返回所有处理后的段落

    def __split_large_parents(self, chunks):
        split_chunks = []
        
        for chunk in chunks:
            if len(chunk.page_content) <= self.__max_parent_size:           # 未超出最大长度的段落，直接添加
                split_chunks.append(chunk)
            else:
                splitter = RecursiveCharacterTextSplitter(                  # 超过最大长度的段落，进行递归分块-----超长块使用递归分块处理
                    chunk_size=self.__max_parent_size,
                    chunk_overlap=config.CHILD_CHUNK_OVERLAP
                )
                sub_chunks = splitter.split_documents([chunk])
                split_chunks.extend(sub_chunks)
        
        return split_chunks

    def __clean_small_chunks(self, chunks):                                         # 处理上一步递归字符分块可能产生的碎片化小块
        cleaned = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < self.__min_parent_size:                    # 对所有小于最小长度的块进行合并处理
                if cleaned:                                                     # 小块在中间，则将其与前一个块进行合并处理
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    for k, v in chunk.metadata.items():
                        if k in cleaned[-1].metadata:
                            cleaned[-1].metadata[k] = f"{cleaned[-1].metadata[k]} -> {v}"
                        else:
                            cleaned[-1].metadata[k] = v
                elif i < len(chunks) - 1:                                       # 小块在开头，则将其与下一个块进行合并处理
                    chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                    for k, v in chunk.metadata.items():                         # 合并小块和后一个块的元数据
                        if k in chunks[i + 1].metadata:
                            chunks[i + 1].metadata[k] = f"{v} -> {chunks[i + 1].metadata[k]}"
                        else:
                            chunks[i + 1].metadata[k] = v
                else:                                                           # 只有一个小块，则将其添加到结果列表中（强制保留小块）
                    cleaned.append(chunk)
            else:
                cleaned.append(chunk)       # 未小于最小长度的块，直接添加
        
        return cleaned

    def __create_child_chunks(self, all_parent_pairs, all_child_chunks, parent_chunks, doc_path):
        for i, p_chunk in enumerate(parent_chunks):
            parent_id = f"{doc_path.stem}_parent_{i}"
            p_chunk.metadata.update({"source": str(doc_path.stem)+".pdf", "parent_id": parent_id})
            
            all_parent_pairs.append((parent_id, p_chunk))
            all_child_chunks.extend(self.__child_splitter.split_documents([p_chunk]))     # 在父块的基础上通过递归字符分块获取子块