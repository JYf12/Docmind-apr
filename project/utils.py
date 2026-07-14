import os
import shutil
import config
import pymupdf.layout
import pymupdf4llm
from pathlib import Path
import glob
import tiktoken
from parsers import get_parser_for_file


def clear_directory_contents(directory: Path) -> None:
    """Delete everything under directory but not the directory itself (safe for Docker volume / bind mount roots)."""
    directory = Path(directory)
    if not directory.is_dir():
        return
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pdf_to_markdown(pdf_path, output_dir):
    doc = pymupdf.open(pdf_path)
    md = pymupdf4llm.to_markdown(doc, header=False, footer=False, page_separators=True, ignore_images=True, write_images=False, image_path=None)
    md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
    output_path = Path(output_dir) / Path(doc.name).stem
    Path(output_path).with_suffix(".md").write_bytes(md_cleaned.encode('utf-8'))

def pdfs_to_markdowns(path_pattern, overwrite: bool = False):
    """将pdf文件转换为markdown文件"""
    output_dir = Path(config.MARKDOWN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in map(Path, glob.glob(path_pattern)):
        md_path = (output_dir / pdf_path.stem).with_suffix(".md")
        if overwrite or not md_path.exists():
            pdf_to_markdown(pdf_path, output_dir)

def convert_to_markdown(file_path, output_dir):
    """通用文档转 Markdown，根据文件类型和 config.PDF_PARSER 自动选择解析器。

    - PDF 文件：由 config.PDF_PARSER 决定使用 pymupdf4llm 还是 markitdown
    - 其他格式（docx/pptx/xlsx 等）：统一使用 markitdown

    Args:
        file_path: 输入文件路径
        output_dir: 输出目录

    Returns:
        生成的 .md 文件路径

    Raises:
        ValueError: 文件格式不被任何解析器支持
    """
    parser = get_parser_for_file(file_path)
    if parser is None:
        raise ValueError(f"Unsupported file format: {file_path}")
    return parser.convert(file_path, output_dir)

def estimate_context_tokens(messages: list) -> int:
    """计算token数"""
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoding.encode(str(msg.content))) for msg in messages if hasattr(msg, 'content') and msg.content)
