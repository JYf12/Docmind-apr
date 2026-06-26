from typing import List
from pydantic import BaseModel, Field

class QueryAnalysis(BaseModel):
    """
    结构化LLM对用户问题的分析结果
        ---使LLM以固定的json格式返回分析结果
    - is_clear: 问题是否清晰
    - questions: 重写后的独立问题列表
    - clarification_needed: 如果问题不清晰，需要的澄清说明

    -example
        ----用户输入："UAV-DETR 相比传统 DETR 有什么优势？"
            LLM 返回：
            {
              "is_clear": true,
              "questions": ["UAV-DETR 相比传统 DETR 的性能优势是什么？"],
              "clarification_needed": ""
            }
        ----用户输入："它怎么样？"
            LLM 返回：
            {
              "is_clear": false,
              "questions": [],
              "clarification_needed": "请问您指的是哪个模型或方法？能具体说明一下吗？"
            }
    """
    is_clear: bool = Field(
        description="Indicates if the user's question is clear and answerable."
    )
    questions: List[str] = Field(
        description="List of rewritten, self-contained questions."
    )
    clarification_needed: str = Field(
        description="Explanation if the question is unclear."
    )