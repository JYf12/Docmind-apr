"""基于 Microsoft Markitdown 的多格式文档解析器。

支持格式：
- PDF (.pdf)
- Word (.docx)
- PowerPoint (.pptx)
- Excel (.xlsx, .xls)
- HTML (.html, .htm)
- EPUB (.epub)
- CSV (.csv)
- JSON (.json)
- XML (.xml)
"""

from pathlib import Path

from markitdown import MarkItDown

from parsers.base import BaseParser


class MarkitdownParser(BaseParser):
    """使用 Microsoft Markitdown 将多种文档格式转换为 Markdown。

    基础模式不使用 LLM 图像描述，仅做文本抽取。
    """

    def __init__(self):
        self._md = MarkItDown()

    def convert(self, file_path: str | Path, output_dir: str | Path) -> Path:
        file_path = Path(file_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self._md.convert(str(file_path))
        md_text = result.text_content

        # 输出路径使用源文件名（不含扩展名）作为 .md 文件名
        output_path = output_dir / file_path.stem
        output_path.with_suffix(".md").write_text(md_text, encoding="utf-8")
        return output_path.with_suffix(".md")

    @property
    def supported_extensions(self) -> list[str]:
        return [
            ".pdf",
            ".docx",
            ".pptx",
            ".xlsx",
            ".xls",
            ".html",
            ".htm",
            ".epub",
            ".csv",
            ".json",
            ".xml",
        ]
