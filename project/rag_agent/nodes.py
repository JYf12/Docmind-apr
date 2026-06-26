from typing import Literal, Set
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage, ToolMessage
from langgraph.types import Command
from .graph_state import State, AgentState
from .schemas import QueryAnalysis
from .prompts_cn import *
# from .prompts_en import *
from utils import estimate_context_tokens
from config import BASE_TOKEN_THRESHOLD, TOKEN_GROWTH_FACTOR

def summarize_history(state: State, llm):
    """总结历史会话[当历史会话记录大于等于4时触发]
    keys:
        --只筛选人类和助手的对话消息
        --只对筛选后的消息取最近6条-----[保持系统的效率和经济性]
        --将最相关的6条人类与AI助手的对话送给llm进行摘要总结
    """
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}
    
    relevant_msgs = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {"conversation_summary": ""}
    
    conversation = "Conversation history:\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

    summary_response = llm.with_config(temperature=0.2).invoke([SystemMessage(content=get_conversation_summary_prompt()), HumanMessage(content=conversation)])
    return {"conversation_summary": summary_response.content, "agent_answers": [{"__reset__": True}]}       # {"__reset__": True}--> 清空助手的消息记录

def rewrite_query(state: State, llm):
    """
    查询重写
    output:
        --问题不明确
        --一个或多个[最多三个]重写的、独立的、适用于文档检索的查询
    keys:
        查询澄清会删除除系统消息外的所有信息，但不会完全丢失上下文，因为在每一次对话发起时（同样包括不清晰的查询）都会总结之前的对话摘要，llm会考虑这些上下文信息以回复当前用户查询
        但这是一种有损压缩操作，每次llm生成清晰查询时都会删除过往对话记录，即便有对话摘要，但仍可能在多次总结过程中丢失重要信息，并且系统无法追溯原始对话
        可选优化方案：只删除导致澄清的那一轮对话，保留之前的有效对话
    """
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")        # 历史会话摘要

    context_section = (f"Conversation Context:\n{conversation_summary}\n" if conversation_summary.strip() else "") + f"User Query:\n{last_message.content}\n"

    llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(QueryAnalysis)
    response = llm_with_structure.invoke([SystemMessage(content=get_rewrite_query_prompt()), HumanMessage(content=context_section)])

    if response.questions and response.is_clear:            # 不需要澄清查询
        delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]        # 欠妥--在当前设置下，小于4轮对话的记录不会被压缩；且直接删除对话记录，仅靠压缩内容回顾上下文有待商榷
        return {"questionIsClear": True, "messages": delete_all, "originalQuery": last_message.content, "rewrittenQuestions": response.questions}

    clarification = response.clarification_needed if response.clarification_needed and len(response.clarification_needed.strip()) > 10 else "I need more information to understand your question."
    return {"questionIsClear": False, "messages": [AIMessage(content=clarification)]}

def request_clarification(state: State):
    """请求澄清节点 - 作为中断点，等待用户提供更清晰的问题"""
    # 此节点主要作为中断点使用，实际澄清消息已在 rewrite_query 中生成（59-60行）--用户会在中断发生之后输入澄清信息，用户消息会被追加了消息列表中，随后针对最新的这条用户消息再进行查询重写，直至不需要澄清
    # 返回空字典表示不修改状态，仅作为流程控制标记
    return {}

# --- Agent Nodes ---
def orchestrator(state: AgentState, llm_with_tools):
    context_summary = state.get("context_summary", "").strip()
    sys_msg = SystemMessage(content=get_orchestrator_prompt())
    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary else []
    )

    # 检测是否刚收到 search_child_chunks 的结果，但未调用 retrieve_parent_chunks
    needs_parent_retrieval = False
    parent_ids_to_retrieve = []

    if state.get("messages"):
        messages = state["messages"]

        # 查找最后一条 ToolMessage
        last_tool_msg = None
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                last_tool_msg = msg
                break

        # 查找最后一条 AIMessage 的 tool_calls
        last_ai_msg = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai_msg = msg
                break

        # 如果最后一条是 ToolMessage 且来自 search_child_chunks
        # 且最后一条 AIMessage 没有调用 retrieve_parent_chunks
        if last_tool_msg and last_tool_msg.name == "search_child_chunks":
            if not last_ai_msg or not any(
                    tc.get("name") == "retrieve_parent_chunks"
                    for tc in getattr(last_ai_msg, "tool_calls", [])
            ):
                # 从搜索结果中提取 Parent IDs
                import re
                content = last_tool_msg.content
                if content != "NO_RELEVANT_CHUNKS":
                    parent_ids = re.findall(r'Parent ID:\s*([^\n]+)', content)
                    parent_ids = [pid.strip() for pid in parent_ids if pid.strip()]

                    # 过滤已检索的 Parent IDs
                    existing_parent_ids = set()
                    if context_summary:
                        existing_matches = re.findall(r'parent::([^\s\n]+)', context_summary)
                        existing_parent_ids.update(existing_matches)

                    # 也检查消息历史中已调用的 retrieve_parent_chunks
                    for msg in messages:
                        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                            for tc in msg.tool_calls:
                                if tc["name"] == "retrieve_parent_chunks":
                                    pid = tc["args"].get("parent_id", "")
                                    if pid:
                                        existing_parent_ids.add(pid)

                    # parent_ids_to_retrieve = [pid for pid in parent_ids if pid not in existing_parent_ids]
                    parent_ids_to_retrieve = list(set([pid for pid in parent_ids if pid not in existing_parent_ids]))
                    if parent_ids_to_retrieve:
                        needs_parent_retrieval = True

    if not state.get("messages"):               # 首次执行
        human_msg = HumanMessage(content=state["question"])
        force_search = HumanMessage(content="要回答这个问题，您必须首先调用“search_child_chunks”。")
        response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search])
        return {"messages": [human_msg, response], "tool_call_count": len(response.tool_calls or []), "iteration_count": 1}
    # 第二次及之后执行
    # response = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
    messages_to_send = [sys_msg] + summary_injection + state["messages"]

    # 如果检测到需要检索父块，添加强制指令
    if needs_parent_retrieval:
        force_instruction = HumanMessage(
            content=f"【系统强制指令】检测到上一步搜索结果的 Parent ID: {', '.join(parent_ids_to_retrieve)}。\n"
                    f"你必须立即调用 'retrieve_parent_chunks' 工具获取这些父文档的完整内容。\n"
                    f"请对以下每个 Parent ID 调用 retrieve_parent_chunks: {', '.join(parent_ids_to_retrieve)}"
        )
        messages_to_send.append(force_instruction)

    response = llm_with_tools.invoke(messages_to_send)

    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {"messages": [response], "tool_call_count": len(tool_calls) if tool_calls else 0, "iteration_count": 1}

def fallback_response(state: AgentState, llm):
    """
    降级响应
        通过上下文摘要和工具调用结果生成最终答案
    """
    seen = set()
    unique_contents = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:            # 添加通过检索工具返回的数据
            unique_contents.append(m.content)
            seen.add(m.content)

    context_summary = state.get("context_summary", "").strip()

    context_parts = []
    if context_summary:
        context_parts.append(f"## Compressed Research Context (from prior iterations)\n\n{context_summary}")
    if unique_contents:
        context_parts.append(
            "## Retrieved Data (current iteration)\n\n" +
            "\n\n".join(f"--- DATA SOURCE {i} ---\n{content}" for i, content in enumerate(unique_contents, 1))
        )

    context_text = "\n\n".join(context_parts) if context_parts else "No data was retrieved from the documents."

    prompt_content = (
        f"USER QUERY: {state.get('question')}\n\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\nProvide the best possible answer using only the data above."
    )
    response = llm.invoke([SystemMessage(content=get_fallback_response_prompt()), HumanMessage(content=prompt_content)])
    return {"messages": [response]}

def should_compress_context(state: AgentState) -> Command[Literal["compress_context", "orchestrator"]]:
    messages = state["messages"]

    new_ids: Set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "retrieve_parent_chunks":
                    raw = tc["args"].get("parent_id") or tc["args"].get("id") or tc["args"].get("ids") or []
                    if isinstance(raw, str):
                        new_ids.add(f"parent::{raw}")
                    else:
                        new_ids.update(f"parent::{r}" for r in raw)

                elif tc["name"] == "search_child_chunks":
                    query = tc["args"].get("query", "")
                    if query:
                        new_ids.add(f"search::{query}")
            break

    updated_ids = state.get("retrieval_keys", set()) | new_ids

    current_token_messages = estimate_context_tokens(messages)                                                  # 计算当前上下文大小
    current_token_summary = estimate_context_tokens([HumanMessage(content=state.get("context_summary", ""))])   # 计算上下文摘要的token数
    current_tokens = current_token_messages + current_token_summary

    max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)

    goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
    return Command(update={"retrieval_keys": updated_ids}, goto=goto)

def compress_context(state: AgentState, llm):
    """
    压缩工具的调用结果
        --根据历史摘要[用户查询+上一轮摘要+历史工具调用和执行情况]llm进行压缩
        --压缩信息添加已执行的工具调用信息
        --更新历史摘要并删除历史消息以压缩上下文
    """
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()

    if not messages:
        return {}

    conversation_text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"
    if existing_summary:
        conversation_text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"

    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            tool_calls_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                tool_calls_info = f" | Tool calls: {calls}"
            conversation_text += f"[ASSISTANT{tool_calls_info}]\n{msg.content or '(tool call only)'}\n\n"
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            conversation_text += f"[TOOL RESULT — {tool_name}]\n{msg.content}\n\n"

    summary_response = llm.invoke([SystemMessage(content=get_context_compression_prompt()), HumanMessage(content=conversation_text)])
    new_summary = summary_response.content

    retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))
        search_queries = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))

        block = "\n\n---\n**Already executed (do NOT repeat):**\n"
        if parent_ids:
            block += "Parent chunks retrieved:\n" + "\n".join(f"- {p.replace('parent::', '')}" for p in parent_ids) + "\n"
        if search_queries:
            block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"
        new_summary += block

    return {"context_summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in messages[1:]]}

def collect_answer(state: AgentState):
    last_message = state["messages"][-1]
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls     # 判断是否为有效的答案[AI的最后回复+内容不为空+不存在工具调用]
    answer = last_message.content if is_valid else "Unable to generate an answer."
    return {
        "final_answer": answer,
        "agent_answers": [{"index": state["question_index"], "question": state["question"], "answer": answer}]
    }
# --- End of Agent Nodes---

def aggregate_answers(state: State, llm):
    """
    聚合答案
        --将所有子问题的答案通过llm进行聚合
        --返回最终答案
    """
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers were generated.")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += (f"\nAnswer {i}:\n"f"{ans['answer']}\n")

    user_message = HumanMessage(content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}""")
    synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt()), user_message])
    return {"messages": [AIMessage(content=synthesis_response.content)]}