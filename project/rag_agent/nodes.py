from typing import Literal, Set
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage, ToolMessage
from langgraph.types import Command
from .graph_state import State, AgentState, CompressionRecord, ContextMetrics
from .schemas import QueryAnalysis
from .prompts_cn import *
# from .prompts_en import *
from utils import estimate_context_tokens
from config import (
    BASE_TOKEN_THRESHOLD, TOKEN_GROWTH_FACTOR,
    MIN_SUMMARIZE_ROUNDS, SUMMARIZE_TOKEN_THRESHOLD, MAX_SUMMARIZE_ROUNDS,
    COMPRESSION_LEVEL_1_RATIO,
    KEEP_RECENT_MSG_COUNT,
)
import config
import re

def summarize_history(state: State, llm):
    """总结历史会话[累计摘要 + 保留最近 N 条，在 aggregate_answers 之后执行]

    压缩策略：
        - 将 messages[1:-N]（跳过 SystemMessage，保留最近 N 条）压缩为累计摘要
        - 摘要以 HumanMessage 注入，包含 [STRUCTURED CONTEXT SUMMARY] + [RECENT CONVERSATION]
        - 删除所有非系统消息，压缩后仅剩 [Sys] + [HumanMsg(摘要+最近N条)]

    triggers:
        - token 阈值超过 SUMMARIZE_TOKEN_THRESHOLD (4000)
        - 消息轮数超过 MAX_SUMMARIZE_ROUNDS (6)
    """
    messages = state["messages"]
    print(f"主图当前消息列表:")
    for msg in messages:
        print(f"  - {type(msg).__name__}: {msg.content[:20]}{'...' if len(msg.content) > 20 else ''}")
    msg_count = len(messages)
    N = KEEP_RECENT_MSG_COUNT

    # 不满足最低轮数时，不压缩
    if msg_count < MIN_SUMMARIZE_ROUNDS:
        return {}

    # Token 估算（排除 SystemMessage）
    non_system_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
    current_tokens = estimate_context_tokens(non_system_msgs)

    # 双重触发条件
    token_exceeded = current_tokens > SUMMARIZE_TOKEN_THRESHOLD
    round_exceeded = msg_count >= MAX_SUMMARIZE_ROUNDS
    if not token_exceeded and not round_exceeded:
        print(f"History summary not triggered: token_exceeded={token_exceeded}, round_exceeded={round_exceeded}")
        return {}

    trigger = "token" if token_exceeded else "rounds"

    # 边界保护：消息数太少不压缩
    if msg_count <= N + 1:
        return {}

    # === 分离消息 ===
    # messages[0]   = SystemMessage
    # messages[:-N] = 待压缩的旧消息
    # messages[-N:]  = 最近 N 条（保留原文）
    old_messages = messages[:-N]
    recent_msgs = messages[-N:]

    # === 构建 LLM 压缩输入（支持增量合并） ===

    previous_summary = state.get("structured_summary", "")
    conversation_text = ""
    if previous_summary.strip():
        conversation_text += f"## [PREVIOUS STRUCTURED SUMMARY]\n{previous_summary}\n\n"
    conversation_text += "## [TO BE SUMMARIZED CONVERSATION]\n"
    for msg in old_messages:
        if isinstance(msg, HumanMessage):
            conversation_text += f"[用户]: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            conversation_text += f"[助手]: {msg.content or '(tool call only)'}\n"

    # === LLM 生成累计摘要 ===
    summary_response = llm.with_config(temperature=0.2).invoke([
        SystemMessage(content=get_structured_summary_prompt()),
        HumanMessage(content=conversation_text)
    ])
    new_structured_summary = summary_response.content

    # === 构建最近 N 条消息的原文文本 ===
    recent_text = ""
    for msg in recent_msgs:
        if isinstance(msg, HumanMessage):
            recent_text += f"[用户]: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            recent_text += f"[助手]: {msg.content or ''}\n"

    # === 注入 HumanMessage（累计摘要 + 最近 N 条原文）===
    summary_injection = HumanMessage(
        content=(
            f"[STRUCTURED CONTEXT SUMMARY]\n\n{new_structured_summary}\n\n"
            f"---\n"
            f"[RECENT CONVERSATION]\n\n{recent_text.strip()}"
        )
    )

    # === 删除所有非系统消息 ===
    remove_ids = [RemoveMessage(id=m.id) for m in messages]
    new_messages = [summary_injection] + remove_ids

    # === 指标采集 ===
    new_summary_tokens = estimate_context_tokens([summary_injection])
    metrics: ContextMetrics = state.get("context_metrics", ContextMetrics())
    metrics.main_compression_count += 1
    metrics.main_tokens_before_total += current_tokens
    metrics.main_tokens_after_total += new_summary_tokens
    metrics.main_trigger_reason = trigger

    print(f"History summary triggered by {trigger}: {new_structured_summary}--new summary tokens: {new_summary_tokens}, old tokens: {current_tokens}")

    return {
        "structured_summary": new_structured_summary,
        "messages": new_messages,
        "agent_answers": [{"__reset__": True}],
        "context_metrics": metrics,
    }

def _extract_recent_conversation(messages, max_turns=4):
    """从 messages 中提取近期对话历史（排除最后一条当前用户消息）

    仅提取 HumanMessage 和 AIMessage（跳过 SystemMessage / ToolMessage），
    取最近 max_turns 轮用户-助手交互，用于在 structured_summary 尚未生成时
    为 rewrite_query 提供指代消解所需的上下文。
    """
    # 排除最后一条（当前用户查询）以及 SystemMessage / ToolMessage
    history_msgs = []
    for msg in messages[:-1]:
        if isinstance(msg, (HumanMessage, AIMessage)):
            history_msgs.append(msg)

    # 只保留最近 max_turns 轮（每轮 = 1 Human + 1 AI = 2 条）
    if len(history_msgs) > max_turns * 2:
        history_msgs = history_msgs[-(max_turns * 2):]

    lines = []
    for msg in history_msgs:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = msg.content or ""
        # AIMessage 内容可能很长（含工具调用结果），截取前 500 字符作为摘要参考
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"[{role}]: {content}")

    return "\n".join(lines)


def rewrite_query(state: State, llm):
    """
    查询重写（作为 START 后第一个节点）
    output:
        --问题不明确
        --一个或多个[最多三个]重写的、独立的、适用于文档检索的查询
    keys:
        - 优先使用 structured_summary 作为对话上下文
        - 若 structured_summary 为空，则从 messages 中提取近期对话历史作为补充上下文
    """
    last_message = state["messages"][-1]
    structured_summary = state.get("structured_summary", "")

    # 上下文 = 累计摘要 或 近期对话历史 + 当前查询【互斥】
    context_section = ""
    if structured_summary.strip():
        context_section += f"## [Conversation Context]\n{structured_summary}\n\n"
    else:
        # 从 messages 中提取近期对话历史（排除最后一条即当前用户消息）
        recent_history = _extract_recent_conversation(state["messages"])
        if recent_history.strip():
            context_section += f"## [Recent Conversation History]\n{recent_history}\n\n"
    context_section += f"## [User Query]\n{last_message.content}\n"

    llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(QueryAnalysis)
    response = llm_with_structure.invoke([SystemMessage(content=get_rewrite_query_prompt()), HumanMessage(content=context_section)])

    if response.questions and response.is_clear:
        return {
            "questionIsClear": True,
            "originalQuery": last_message.content,
            "rewrittenQuestions": response.questions,
            "agent_answers": [{"__reset__": True}],   # 新查询开始前清空上一轮的答案，防止多轮对话状态污染
        }

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

    # 检测是否刚收到 search_child_chunks 的结果，但未调用 retrieve_parent_chunks 或 retrieve_images
    needs_parent_retrieval = False
    parent_ids_to_retrieve = []
    needs_image_retrieval = False
    image_ids_to_retrieve = []

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

                # 同样提取 Image IDs（图片 Summary 片段）
                image_ids = re.findall(r'Image ID:\s*([^\n]+)', content)
                image_ids = [iid.strip() for iid in image_ids if iid.strip()]
                if image_ids:
                    # 过滤已检索的 Image IDs
                    existing_image_ids = set()
                    if context_summary:
                        existing_img_matches = re.findall(r'image::([^\s\n]+)', context_summary)
                        existing_image_ids.update(existing_img_matches)
                    for msg in messages:
                        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                            for tc in msg.tool_calls:
                                if tc["name"] == "retrieve_images":
                                    raw = tc["args"].get("image_ids", "")
                                    if isinstance(raw, str):
                                        existing_image_ids.add(raw)
                                    elif isinstance(raw, list):
                                        existing_image_ids.update(raw)

                    image_ids_to_retrieve = list(set([iid for iid in image_ids if iid not in existing_image_ids]))
                    if image_ids_to_retrieve:
                        needs_image_retrieval = True

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

    # 如果检测到需要检索图片元数据，添加强制指令
    if needs_image_retrieval:
        force_image_instruction = HumanMessage(
            content=f"【系统强制指令】检测到上一步搜索结果的 Image ID: {', '.join(image_ids_to_retrieve)}。\n"
                    f"你必须立即调用 'retrieve_images' 工具获取这些图片的元数据和描述。\n"
                    f"请调用 retrieve_images 并传入 image_ids: {', '.join(image_ids_to_retrieve)}"
        )
        messages_to_send.append(force_image_instruction)

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

# ===== 关键信息兜底工具函数 =====

def extract_file_sources(messages: list) -> set:
    """从消息中提取所有文件来源名"""
    sources = set()
    for msg in messages:
        if isinstance(msg, (AIMessage, ToolMessage)) and msg.content:
            content = str(msg.content)
            # 匹配文件名（xxx.pdf, xxx.md, xxx.docx 等）— 使用 ASCII 字符集避免匹配中文
            files = re.findall(r'[a-zA-Z0-9_\-]+\.(?:pdf|md|docx|txt)', content)
            sources.update(files)
            # 匹配中文引导的来源
            cn_files = re.findall(r'(?:来源|来自|文件)[：:]*\s*(\S+\.\w+)', content)
            sources.update(cn_files)
    return sources


def verify_and_patch_summary(summary: str, critical_sources: set) -> tuple[str, int]:
    """检查关键来源是否已保留，返回(补丁后的摘要, 保留数)"""
    if not critical_sources:
        return summary, 0
    missing = [s for s in critical_sources if s not in summary]
    if not missing:
        return summary, len(critical_sources)
    preserved = len(critical_sources) - len(missing)
    summary += f"\n\n## 保留数据\n"
    summary += "\n".join(f"- 来源文件: {s}" for s in sorted(missing))
    return summary, preserved


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

                elif tc["name"] == "retrieve_images":
                    raw = tc["args"].get("image_ids") or tc["args"].get("image_id") or []
                    if isinstance(raw, str):
                        new_ids.add(f"image::{raw}")
                    elif isinstance(raw, list):
                        new_ids.update(f"image::{r}" for r in raw)
            break

    updated_ids = state.get("retrieval_keys", set()) | new_ids

    current_token_messages = estimate_context_tokens(messages)
    current_token_summary = estimate_context_tokens([HumanMessage(content=state.get("context_summary", ""))])
    current_tokens = current_token_messages + current_token_summary

    max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)
    ratio = current_tokens / max_allowed if max_allowed > 0 else 1.0

    if ratio <= 1.0:
        print(f"Context within token limit, no compression needed. Ratio: {ratio:.2f}--Current tokens: {current_tokens}, Max allowed: {max_allowed}")
        return Command(update={"retrieval_keys": updated_ids}, goto="orchestrator")

    # 两级压缩等级判定
    # Level 1: 仅删 ToolMessage（占比 ~90%），保留推理链
    # Level 2: 激进删除所有非 System 消息（极端情况兜底）
    level = 1 if ratio <= COMPRESSION_LEVEL_1_RATIO else 2.5

    return Command(
        update={
            "compression_level": level,
            "trigger_ratio": ratio,
            "retrieval_keys": updated_ids,
        },
        goto="compress_context"
    )


def compress_context(state: AgentState, llm):
    """
    分层渐进式压缩 + 关键信息兜底
        --根据历史摘要[用户查询+上一轮摘要+历史工具调用和执行情况]llm进行压缩
        --按压缩等级决定删除哪些消息（轻度/中度/激进）
        --自动验证压缩后文件来源是否保留，缺失则追加
    """
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()
    level = state.get("compression_level", 2)

    if not messages:
        return {}

    # [兜底] 压缩前提取关键数据点
    sources_before = extract_file_sources(messages)

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

    tokens_before = estimate_context_tokens(messages[1:])
    summary_response = llm.invoke([
        SystemMessage(content=get_context_compression_prompt()),
        HumanMessage(content=conversation_text)
    ])
    new_summary = summary_response.content

    # [兜底] 验证关键来源是否保留
    patched_summary, sources_preserved = verify_and_patch_summary(new_summary, sources_before)

    # 追加已执行信息
    retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))
        image_ids = sorted(r for r in retrieved_ids if r.startswith("image::"))
        search_queries = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))

        block = "\n\n---\n**Already executed (do NOT repeat):**\n"
        if parent_ids:
            block += "Parent chunks retrieved:\n" + "\n".join(f"- {p.replace('parent::', '')}" for p in parent_ids) + "\n"
        if image_ids:
            block += "Image metadata retrieved:\n" + "\n".join(f"- {i.replace('image::', '')}" for i in image_ids) + "\n"
        if search_queries:
            block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"
        patched_summary += block

    final_summary = patched_summary

    # 按等级决定删除哪些消息
    if level == 1:
        # Level 1: 仅删除 ToolMessage（占比 ~90%），保留推理链完整
        messages_to_remove = [
            RemoveMessage(id=m.id) for m in messages[1:]
            if isinstance(m, ToolMessage) or (isinstance(m, AIMessage) and getattr(m, "tool_calls", None))
        ]
    else:
        # Level 2: 删除所有非 System 消息（极端情况兜底）
        messages_to_remove = [RemoveMessage(id=m.id) for m in messages[1:]]

    tokens_after = estimate_context_tokens([HumanMessage(content=final_summary)])
    tokens_removed = tokens_before - tokens_after

    print(f"Compression level {level}: {tokens_removed} tokens removed ({tokens_before} -> {tokens_after})")

    # 采集子图指标
    record = CompressionRecord(
        level=level,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_removed=tokens_removed,
        sources_before=len(sources_before),
        sources_preserved=sources_preserved,
        ratio=state.get("trigger_ratio", 0.0),
        summary_length=len(final_summary),
    )

    metrics: ContextMetrics = state.get("sub_context_metrics", ContextMetrics())
    metrics.sub_compression_records.append(record)
    metrics.sub_total_compressions += 1
    metrics.sub_total_tokens_removed += tokens_removed
    metrics.sub_level_counts[level - 1] += 1

    return {
        "context_summary": final_summary,
        "messages": messages_to_remove,
        "sub_context_metrics": metrics,
    }

def collect_answer(state: AgentState):
    last_message = state["messages"][-1]
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls
    answer = last_message.content if is_valid else "Unable to generate an answer."

    # === 1. 先提取检索上下文（在删除消息前完成） ===
    contexts = []
    seen = set()
    image_contexts = []
    image_seen = set()
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and msg.content not in seen:            # 潜在的风险：若测试样本中不包含图片内容的描述GT，此处将图片Summary计入contexts可能会导致上下文召回率及其他指标产生波动
            if not any(kw in msg.content for kw in [
                "NO_RELEVANT_CHUNKS", "NO_PARENT_DOCUMENTS",
                "NO_PARENT_DOCUMENT", "RETRIEVAL_ERROR", "PARENT_RETRIEVAL_ERROR"
            ]):
                contexts.append(msg.content)
                seen.add(msg.content)

        # 收集 retrieve_images 返回的图片元数据（用于最终多模态合成）
        if isinstance(msg, ToolMessage) and msg.name == "retrieve_images":
            if msg.content not in image_seen and msg.content not in (
                "NO_IMAGE_METADATA_FOUND", "IMAGE_RETRIEVAL_ERROR", "IMAGE_STORE_NOT_AVAILABLE"
            ):
                image_contexts.append(msg.content)
                image_seen.add(msg.content)

    # === 2. 清除子图所有消息，阻止 ToolMessage 回流污染主图 ==
    remove_all = [RemoveMessage(id=m.id) for m in state["messages"]]

    # 捎带子图指标到主图的 agent_answers 中（通过 accumulate_or_reset reducer 累积）
    extra = {}
    sub_metrics = state.get("sub_context_metrics")
    if sub_metrics and hasattr(sub_metrics, 'to_dict'):
        extra["context_metrics"] = sub_metrics.to_dict()

    return {
        "final_answer": answer,
        "messages": remove_all,
        "agent_answers": [{
            "index": state["question_index"],
            "question": state["question"],
            "answer": answer,
            "contexts": contexts,       # ← 检索内容走 agent_answers 通道，供评测读取
            "image_contexts": image_contexts,  # ← 图片元数据，供 aggregate_answers 多模态合成
            **extra,
        }]
    }
# --- End of Agent Nodes---

def aggregate_answers(state: State, llm, multimodal_llm=None):
    """
    聚合答案
        --将所有子问题的答案通过llm进行聚合
        --如果存在图片上下文且 multimodal_llm 可用，启用多模态综合
        --否则降级为纯文本综合（图片 Summary 的文本描述仍然被使用）
        --返回最终答案
    """
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers were generated.")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])
    # print("Aggregating answers:", sorted_answers)

    # 收集所有子问题中的图片上下文，按 image_id 去重
    all_image_contexts = []
    image_ids_seen = set()
    for ans in sorted_answers:
        for img_text in ans.get("image_contexts", []):
            # 从 image_context 文本中提取 image_id 用于去重
            import re
            mids = re.findall(r'Image ID:\s*([^\n]+)', img_text)
            new_ids = [mid.strip() for mid in mids if mid.strip() not in image_ids_seen]
            if new_ids:
                image_ids_seen.update(new_ids)
                all_image_contexts.append(img_text)

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += (f"\nAnswer {i}:\n"f"{ans['answer']}\n")

    # 决定使用多模态 LLM 还是纯文本 LLM（召回文档包括图片Summary即启用多模态生成模型，否则降级为纯文本综合）
    use_multimodal = multimodal_llm is not None and all_image_contexts
    synthesis_llm = multimodal_llm if use_multimodal else llm

    if use_multimodal:
        print("✅ Using multimodal synthesis.")
        # 解析图片路径并从磁盘加载 base64
        from core.image_manager import _encode_image_base64

        # 构建 image_path -> description 映射，支持单个 img_text 包含多个图片块
        all_context_text = "\n".join(all_image_contexts)
        path_to_desc = {}
        for desc_match in re.finditer(
            r'Description:\s*([^\n]+).*?Image Path:\s*([^\n]+)',
            all_context_text, re.DOTALL
        ):
            desc_val = desc_match.group(1).strip()
            path_val = desc_match.group(2).strip()
            if path_val:
                path_to_desc[path_val] = desc_val

        image_paths_to_load = list(path_to_desc.keys())

        # 限制图片数量防止 context 爆炸
        image_paths_to_load = image_paths_to_load[:config.MAX_IMAGES_PER_ANSWER]

        content: list = [
            {"type": "text", "text": (
                f"Original user question: {state['originalQuery']}\n\n"
                f"Retrieved answers from sub-agents:{formatted_answers}\n\n"
                f"Please synthesize a comprehensive answer based on the retrieved answers and the information in the images below.\n"
                f"Note: Each image is preceded by a text label '[Image: ...]' summarizing its content. These labels are auto-generated and may be inaccurate. Always prioritize what you actually see in the image over the text label.\n"
            )}
        ]
        for img_path in image_paths_to_load:
            try:
                img_uri = _encode_image_base64(img_path)
                desc = path_to_desc.get(img_path, Path(img_path).name)[:200]
                content.append({"type": "text", "text": f"\n[Image: {desc}]"})
                content.append({"type": "image_url", "image_url": {"url": img_uri}})
            except Exception as e:
                print(f"  ⚠ Failed to load image {img_path}: {e}")

        user_message = HumanMessage(content=content)
        synthesis_response = synthesis_llm.invoke(
            [SystemMessage(content=get_aggregation_prompt()), user_message]
        )
    else:
        # 纯文本路径（现有行为）
        # 如果有 image_contexts，以文本形式注入供纯文本 LLM 参考
        print("✅ Using text-only synthesis.")
        image_note = ""
        if all_image_contexts:
            image_note = (
                "\n\n---\n**Image descriptions retrieved (no vision model available — text descriptions only):**\n" +
                "\n".join(f"- {ctx[:300]}" for ctx in all_image_contexts[:config.MAX_IMAGES_PER_ANSWER])
            )

        user_message = HumanMessage(content=(
            f"Original user question: {state['originalQuery']}\nRetrieved answers:{formatted_answers}{image_note}"
        ))
        synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt()), user_message])

    # 合并各子图的 metrics 到主图
    merged = None
    main_metrics: ContextMetrics = state.get("context_metrics", ContextMetrics())
    for ans in sorted_answers:
        sub_data = ans.get("context_metrics")
        if sub_data and isinstance(sub_data, dict):
            if merged is None:
                # 转换 dict 回 ContextMetrics
                merged = ContextMetrics()
            sub_data = sub_data.get("sub", {})
            merged.sub_total_compressions += sub_data.get("compressions", 0)
            merged.sub_total_tokens_removed += sub_data.get("tokens_removed", 0)

    if merged:
        main_metrics.sub_total_compressions += merged.sub_total_compressions
        main_metrics.sub_total_tokens_removed += merged.sub_total_tokens_removed
        main_metrics.sub_level_counts = [
            main_metrics.sub_level_counts[i] + merged.sub_level_counts[i]
            for i in range(3)
        ]

    return {"messages": [AIMessage(content=synthesis_response.content)], "context_metrics": main_metrics}