from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from functools import partial

from .graph_state import State
from .nodes import *
from .edges import *

def create_agent_graph(llm, tools_list, multimodal_llm=None):
    llm_with_tools = llm.bind_tools(tools_list)
    tool_node = ToolNode(tools_list)

    checkpointer = InMemorySaver()

    """子图构建---处理单个查询的完整生命周期"""
    print("Compiling agent graph...")
    agent_builder = StateGraph(AgentState)
    agent_builder.add_node("orchestrator", partial(orchestrator, llm_with_tools=llm_with_tools))    # 核心调度器[决定：调用工具/降级响应/生成答案]
    agent_builder.add_node("tools", tool_node)                                                      # 工具节点
    agent_builder.add_node("compress_context", partial(compress_context, llm=llm))                  # 上下文压缩器
    agent_builder.add_node("fallback_response", partial(fallback_response, llm=llm))                # 降级响应器[达到最大迭代次数后，基于已有信息生成最佳答案]
    agent_builder.add_node(should_compress_context)                                                 # 动态决策器[基于token阈值判断是否需要压缩上下文]--边逻辑，非普通节点
    agent_builder.add_node(collect_answer)                                                          # 聚合答案

    agent_builder.add_edge(START, "orchestrator")
    agent_builder.add_conditional_edges("orchestrator", route_after_orchestrator_call, {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"})
    agent_builder.add_edge("tools", "should_compress_context")
    agent_builder.add_edge("compress_context", "orchestrator")
    agent_builder.add_edge("fallback_response", "collect_answer")
    agent_builder.add_edge("collect_answer", END)

    agent_subgraph = agent_builder.compile()

    """主图构建---协调整个问答流程"""
    graph_builder = StateGraph(State)
    graph_builder.add_node("summarize_history", partial(summarize_history, llm=llm))                # 聊天历史压缩器[回答完成后，将历史对话压缩为累计摘要+保留最近N条消息]---压缩上下文并安全删除旧消息
    graph_builder.add_node("rewrite_query", partial(rewrite_query, llm=llm))                        # 查询重写器[将用户输入进行重写【将用户输入进行重写，以适应AI的输入格式】]
    graph_builder.add_node(request_clarification)                                                   # 请求澄清器[当查询无法得到有效答案时，请求澄清]
    graph_builder.add_node("agent", agent_subgraph)                                                 # 问答子图[处理单个查询的完整生命周期]--对每个重写的子问题并行执行智能代理流程
    graph_builder.add_node("aggregate_answers", partial(aggregate_answers, llm=llm, multimodal_llm=multimodal_llm))  # 聚合答案器[将多个子问题的答案聚合为最终答案，支持多模态]

    graph_builder.add_edge(START, "rewrite_query")                                                  # START → rewrite_query（rewrite_query 可直接获取 messages[-1] 作为当前查询）
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)                       # 触发路由判断：问题清晰->agent | 否则->request_clarification
    graph_builder.add_edge("request_clarification", "rewrite_query")                                # 澄清-重写循环[直至问题清晰并交由agent处理]
    graph_builder.add_edge(["agent"], "aggregate_answers")                                          # 聚合所有子图的答案
    graph_builder.add_edge("aggregate_answers", "summarize_history")                                # 回答完成后 → 压缩历史对话
    graph_builder.add_edge("summarize_history", END)                                                # 压缩完成 → 结束

    agent_graph = graph_builder.compile(checkpointer=checkpointer, interrupt_before=["request_clarification"])      # inerrupt_before: 触发中断的节点---中断机制--在需求澄清前中断[只有路由到request_clarification的节点时才会触发中断]

    print("✓ Agent graph compiled successfully.")
    return agent_graph