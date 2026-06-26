from typing import Literal
from langgraph.types import Send
from .graph_state import State, AgentState
from config import MAX_ITERATIONS, MAX_TOOL_CALLS

def route_after_rewrite(state: State) -> Literal["request_clarification", "agent"]:
    """
    对查询重写器进行动态路由判断

    -request_clarification：请求澄清
    -agent：处理单个查询的完整生命周期
    """
    if not state.get("questionIsClear", False):     # 查询不明确时路由到request_clarification节点
        return "request_clarification"
    else:                                           # 查询明确时路由到agent节点(使用Send对象并行启动多个agent实例，每个agent独立处理一个重写后的问题)
        """并行发送多个请求
        每个Send对象会启动一个独立的agent节点实例来处理对应的问题
        """
        return [
                Send("agent", {"question": query, "question_index": idx, "messages": []})
                for idx, query in enumerate(state["rewrittenQuestions"])
            ]
    
def route_after_orchestrator_call(state: AgentState) -> Literal["tool", "fallback_response", "collect_answer"]:
    """
    对子图的核心调度器进行动态路由判断
    -tool：工具调用
    -fallback_response：降级响应
    -collect_answer：生成答案
    """
    iteration = state.get("iteration_count", 0)
    tool_count = state.get("tool_call_count", 0)

    if iteration >= MAX_ITERATIONS or tool_count > MAX_TOOL_CALLS:          # agent迭代次数超限/tool调用次数超限---->降级响应
        return "fallback_response"

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        return "collect_answer"
    
    return "tools"