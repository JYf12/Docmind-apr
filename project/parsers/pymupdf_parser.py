"""基于 pymupdf4llm 的 PDF 解析器。

将现有 utils.py 中 pdf_to_markdown() 的核心逻辑搬迁至此，
保持完全一致的参数和行为。
"""

from pathlib import Path

import pymupdf
import pymupdf4llm

import config
from parsers.base import BaseParser


class PymupdfParser(BaseParser):
    """使用 pymupdf4llm 将 PDF 转换为 Markdown。

    参数与原有 utils.pdf_to_markdown() 完全一致：
    - header=False, footer=False：去除页眉页脚
    - page_separators=True：插入分页标记
    - ignore_images=True：丢弃图片
    """

    def convert(self, file_path: str | Path, output_dir: str | Path) -> Path:
        doc = pymupdf.open(str(file_path))
        """md = pymupdf4llm.to_markdown(
            doc,
            header=False,
            footer=False,
            page_separators=True,   # PDF 中图片如果带有 OCR 文本层（如扫描件），pymupdf4llm 会提取这些文字
            ignore_images=False,    # 设置 write_images=True + image_path，将 PDF 中的图片导出为 png/jpg 文件，并在 Markdown 中插入 ![](path) 引用
            write_images=False,
            image_path=None,
        )"""

        md = pymupdf4llm.to_markdown(
            doc,
            header=False,
            footer=False,
            page_separators=True,  # PDF 中图片如果带有 OCR 文本层（如扫描件），pymupdf4llm 会提取这些文字
            ignore_images=False,    # 设置 write_images=True + image_path，将 PDF 中的图片导出为 png/jpg 文件，并在 Markdown 中插入 ![](path) 引用
            write_images=True,
            image_path=config.MARKDOWN_IMAGES_DIR,
            # table_strategy="lines",     # 显式指定表格检测策略
            # margins=(0, 0, 0, 0)  # 保留边缘表格
        )
        md_cleaned = (
            md.encode("utf-8", errors="surrogatepass")
            .decode("utf-8", errors="ignore")
        )
        output_path = Path(output_dir) / Path(doc.name).stem
        output_path.with_suffix(".md").write_bytes(md_cleaned.encode("utf-8"))
        return output_path.with_suffix(".md")

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]
