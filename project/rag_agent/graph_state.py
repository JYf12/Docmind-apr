from typing import List, Annotated, Set
from langgraph.graph import MessagesState
import operator
from dataclasses import dataclass, field

"""
自定义reducer--状态合并函数
"""
def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    if new and any(item.get('__reset__') for item in new):
        return []               # 重置状态（清空列表）
    return existing + new       # 追加消息

def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    return a | b

# ===== 指标系统 =====

@dataclass
class CompressionRecord:
    """单次压缩事件的记录"""
    level: int                          # 压缩等级 1-3
    tokens_before: int                  # 压缩前总 token 数
    tokens_after: int                   # 压缩后（摘要）token 数
    tokens_removed: int                 # 删除的消息 token 数
    sources_before: int                 # 压缩前消息中的来源文件数
    sources_preserved: int              # 压缩后摘要中保留的来源文件数
    ratio: float                        # current_tokens / max_allowed（触发时的比例）
    summary_length: int                 # 生成的摘要长度（字符数）


@dataclass
class ContextMetrics:
    """上下文管理全周期指标"""
    # ---- 主图指标 ----
    main_compression_count: int = 0     # 主图压缩次数
    main_tokens_before_total: int = 0   # 主图累计压缩前 token
    main_tokens_after_total: int = 0    # 主图累计压缩后 token
    main_trigger_reason: str = ""       # 最近一次触发原因（"token" / "rounds" / ""）

    # ---- 子图指标 ----
    sub_compression_records: list[CompressionRecord] = field(default_factory=list)
    sub_total_compressions: int = 0      # 子图压缩总次数
    sub_total_tokens_removed: int = 0    # 子图累计删除 token 数
    sub_level_counts: list[int] = field(default_factory=lambda: [0, 0, 0])  # 各级压缩次数

    # ---- 收益汇总（只读属性） ----
    @property
    def main_total_tokens_saved(self) -> int:
        return max(0, self.main_tokens_before_total - self.main_tokens_after_total)

    @property
    def sub_total_tokens_saved(self) -> int:
        return self.sub_total_tokens_removed

    @property
    def total_tokens_saved(self) -> int:
        return self.main_total_tokens_saved + self.sub_total_tokens_saved

    @property
    def overall_compression_ratio(self) -> float:
        total_before = self.main_tokens_before_total + self.sub_total_tokens_removed
        total_after = self.main_tokens_after_total
        if total_before == 0:
            return 0.0
        return round((1 - total_after / total_before) * 100, 1)

    @property
    def avg_information_retention_rate(self) -> float:
        records = self.sub_compression_records
        if not records:
            return 100.0
        rates = [r.sources_preserved / max(r.sources_before, 1) * 100 for r in records]
        return round(sum(rates) / len(rates), 1) if rates else 100.0

    @property
    def level_distribution(self) -> str:
        return f"L1:{self.sub_level_counts[0]} L2:{self.sub_level_counts[1]} L3:{self.sub_level_counts[2]}"

    def to_dict(self) -> dict:
        return {
            "main": {
                "compressions": self.main_compression_count,
                "tokens_saved": self.main_total_tokens_saved,
                "last_trigger": self.main_trigger_reason,
            },
            "sub": {
                "compressions": self.sub_total_compressions,
                "tokens_removed": self.sub_total_tokens_removed,
                "level_distribution": self.level_distribution,
                "avg_info_retention": self.avg_information_retention_rate,
            },
            "summary": {
                "total_tokens_saved": self.total_tokens_saved,
                "compression_ratio_pct": self.overall_compression_ratio,
            }
        }

class State(MessagesState):
    """State for main agent graph  管理整个问答流程的全局状态"""
    questionIsClear: bool = False
    conversation_summary: str = ""              # [已废弃] 不再写入，保留字段避免 breaking change
    structured_summary: str = ""                 # [新增] summarize_history 生成的累计结构化摘要
    originalQuery: str = ""
    rewrittenQuestions: List[str] = []
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []      # 用于收集多个子问题的答案，使用自定义reducer进行状态合并，同名字段才会从子图回传到主图
    context_metrics: ContextMetrics = field(default_factory=ContextMetrics)  # 主图+子图汇总指标

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
    compression_level: int = 0                                       # 0=不压缩, 1=轻度, 2=中度, 3=激进
    trigger_ratio: float = 0.0                                       # 触发压缩时的 tokens/max_allowed 比例
    sub_context_metrics: ContextMetrics = field(default_factory=ContextMetrics)  # 子图指标（作用域仅在子图内部）