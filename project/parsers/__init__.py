"""文档解析器模块 — 提供统一的多格式文档转 Markdown 接口。

使用方式:
    from parsers import get_parser_for_file, get_pdf_parser
    parser = get_parser_for_file("doc.pdf")
    md_path = parser.convert("doc.pdf", "output_dir/")
"""

from pathlib import Path

import config
from parsers.base import BaseParser
from parsers.pymupdf_parser import PymupdfParser
from parsers.markitdown_parser import MarkitdownParser


def get_pdf_parser() -> BaseParser:
    """根据 config.PDF_PARSER 返回 PDF 解析器实例。

    Returns:
        PymupdfParser 或 MarkitdownParser
    """
    if config.PDF_PARSER == "markitdown":
        return MarkitdownParser()
    return PymupdfParser()


def get_parser_for_file(file_path: str | Path) -> BaseParser | None:
    """根据文件扩展名返回合适的解析器，不支持的类型返回 None。

    - PDF 文件：根据 config.PDF_PARSER 选择 pymupdf4llm 或 markitdown       （友情提示：Markitdown的纯文本提取方式会丢失pdf中的样式信息，性能不如pymupdf4llm）
    - 其他格式（docx/pptx/xlsx/html/epub 等）：统一使用 Markitdown

    Args:
        file_path: 输入文件路径

    Returns:
        对应的解析器实例，或 None（不支持的类型）
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return get_pdf_parser()

    md_parser = MarkitdownParser()
    if ext in md_parser.supported_extensions:
        return md_parser

    return None
