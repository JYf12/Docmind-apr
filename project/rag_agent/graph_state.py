from typing import List, Annotated, Set
from langgraph.graph import MessagesState
import operator

"""
自定义reducer--状态合并函数
"""
def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    if new and any(item.get('__reset__') for item in new):
        return []               # 重置状态（清空列表）
    return existing + new       # 追加消息

def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    return a | b

class State(MessagesState):
    """State for main agent graph  管理整个问答流程的全局状态"""
    questionIsClear: bool = False
    conversation_summary: str = ""
    originalQuery: str = "" 
    rewrittenQuestions: List[str] = []
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []      # 用于收集多个子问题的答案，使用自定义reducer进行状态合并

class AgentState(MessagesState):
    """State for individual agent subgraph  每个子图agent内节点单独共享"""
    question: str = ""
    question_index: int = 0
    context_summary: str = ""
    retrieval_keys: Annotated[Set[str], set_union] = set()          # Reducer: 集合并
    final_answer: str = ""
    agent_answers: List[dict] = []                                  # Reducer: 覆盖（不指定reducer时，默认使用覆盖策略）
    tool_call_count: Annotated[int, operator.add] = 0               # Reducer: 累加
    iteration_count: Annotated[int, operator.add] = 0               # Reducer: 累加