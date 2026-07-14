"""文档解析器抽象基类。"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseParser(ABC):
    """所有文档解析器的基类。

    子类需实现 convert() 和 supported_extensions 属性。
    """

    @abstractmethod
    def convert(self, file_path: str | Path, output_dir: str | Path) -> Path:
        """将文档转换为 Markdown 并写入文件。

        Args:
            file_path: 输入文件路径
            output_dir: 输出目录

        Returns:
            生成的 .md 文件路径
        """
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """返回该解析器支持的文件扩展名列表（含点号，如 [".pdf"]）。"""
        ...
