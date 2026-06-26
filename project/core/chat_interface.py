import json
import re
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage

SILENT_NODES = {"rewrite_query"}                        # 静默节点：不显示最终输出
SYSTEM_NODES = {"summarize_history", "rewrite_query"}   # 系统节点：显示在折叠框中

SYSTEM_NODE_CONFIG = {
    "rewrite_query":     {"title": "🔍 Query Analysis & Rewriting"},
    "summarize_history": {"title": "📋 Chat History Summary"},
}

# --- Helpers ---

def make_message(content, *, title=None, node=None):
    msg = {"role": "assistant", "content": content}
    if title or node:
        msg["metadata"] = {k: v for k, v in {"title": title, "node": node}.items() if v}        # metadata字段不为空的消息显示在折叠卡片中
    return msg


def find_msg_idx(messages, node):
    return next(                                                                            # 返回第一个匹配项的索引，如果没有匹配项，则返回None
        (i for i, m in enumerate(messages) if m.get("metadata", {}).get("node") == node),
        None,
    )


def parse_rewrite_json(buffer):
    match = re.search(r"\{.*\}", buffer, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None


def format_rewrite_content(buffer):
    data = parse_rewrite_json(buffer)
    if not data:
        return "⏳ Analyzing query..."
    if data.get("is_clear"):
        lines = ["✅ **Query is clear**"]
        if data.get("questions"):
            lines += ["\n**Rewritten queries:**"] + [f"- {q}" for q in data["questions"]]
    else:
        lines = ["❓ **Query is unclear**"]
        clarification = data.get("clarification_needed", "")
        if clarification and clarification.strip().lower() != "no":
            lines.append(f"\nClarification needed: *{clarification}*")
    return "\n".join(lines)

# --- End of Helpers ---

class ChatInterface:

    def __init__(self, rag_system):
        self.rag_system = rag_system

    def _handle_system_node(self, chunk, node, response_messages, system_node_buffer):
        """Update (or create) the collapsible system-node message and surface any clarification."""
        system_node_buffer[node] = system_node_buffer.get(node, "") + chunk.content
        buffer = system_node_buffer[node]                                                       # 累积流式内容到缓冲区
        title  = SYSTEM_NODE_CONFIG[node]["title"]
        content = format_rewrite_content(buffer) if node == "rewrite_query" else buffer

        idx = find_msg_idx(response_messages, node)                                             # 查找是否已存在该节点的消息
        if idx is None:
            response_messages.append(make_message(content, title=title, node=node))             # 没有该节点消息，则创建---创建系统消息，显示在折叠卡片
        else:
            response_messages[idx]["content"] = content                                         # 该节点消息已存在，则更新

        if node == "rewrite_query":
            self._surface_clarification(buffer, response_messages)

    def _surface_clarification(self, buffer, response_messages):
        """If the query is unclear, add/update a plain clarification message."""
        data          = parse_rewrite_json(buffer) or {}
        clarification = data.get("clarification_needed", "")
        if not data.get("is_clear") and clarification.strip().lower() not in ("", "no"):
            cidx = find_msg_idx(response_messages, "clarification")
            if cidx is None:
                response_messages.append(make_message(clarification, node="clarification"))
            else:
                response_messages[cidx]["content"] = clarification

    def _handle_tool_call(self, chunk, response_messages, active_tool_calls):
        """Register new tool calls as collapsible messages."""
        for tc in chunk.tool_calls:
            if tc.get("id") and tc["id"] not in active_tool_calls:
                response_messages.append(
                    make_message(f"Running `{tc['name']}`...", title=f"🛠️ {tc['name']}")                # 创建工具消息，在卡片中显示
                )
                active_tool_calls[tc["id"]] = len(response_messages) - 1

    def _handle_tool_result(self, chunk, response_messages, active_tool_calls):
        """Fill in the tool result inside the matching collapsible message."""
        idx = active_tool_calls.get(chunk.tool_call_id)
        if idx is not None:
            preview = str(chunk.content)[:300]
            suffix  = "\n..." if len(str(chunk.content)) > 300 else ""
            response_messages[idx]["content"] = f"```\n{preview}{suffix}\n```"

    def _handle_llm_token(self, chunk, node, response_messages):
        """Append streaming LLM tokens to the last plain assistant message."""
        last = response_messages[-1] if response_messages else None
        if not (last and last.get("role") == "assistant" and "metadata" not in last):
            response_messages.append(make_message(""))                                                  # 创建没有metadata字段的消息，显示为普通消息
        response_messages[-1]["content"] += chunk.content

    def chat(self, message, history):
        """Generator that streams Gradio chat message dicts."""
        if not self.rag_system.agent_graph:
            yield "⚠️ System not initialized!"
            return

        config = self.rag_system.get_config()
        current_state = self.rag_system.agent_graph.get_state(config)       # 获取当前状态

        try:
            # 在当前对话中存在下一个待执行节点---说明当前处于中断状态
            if current_state.next:                  # 中断澄清-----只有中断触发时才会有next
                self.rag_system.agent_graph.update_state(config, {"messages": [HumanMessage(content=message.strip())]})
                stream_input = None     # 将新的用户查询已通过update_state添加到messages
            else:                                   # 没有下一个待执行节点（此轮对话已经结束） 新对话开始--用户查询通过stream传入
                stream_input = {"messages": [HumanMessage(content=message.strip())]}        # 将用户消息作为 stream 的输入-->graph从START节点开始执行

            response_messages  = []         # 存储所有要显示的消息
            active_tool_calls  = {}         # 追踪正在执行的工具调用 {tool_call_id: message_index}
            system_node_buffer = {}         # 缓冲系统节点的流式输出 {node_name: accumulated_text}

            for chunk, metadata in self.rag_system.agent_graph.stream(stream_input, config=config, stream_mode="messages"):     # stream_mode="messages"：以消息级别流式输出，每个 chunk 是一个消息片段
                node = metadata.get("langgraph_node", "")       # 产生当前消息片段的节点名称

                # 系统节点消息
                if node in SYSTEM_NODES and isinstance(chunk, AIMessageChunk) and chunk.content:
                    self._handle_system_node(chunk, node, response_messages, system_node_buffer)

                # 工具调用
                elif hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    self._handle_tool_call(chunk, response_messages, active_tool_calls)         # 创建工具调用占位符

                # 工具结果
                elif isinstance(chunk, ToolMessage):
                    self._handle_tool_result(chunk, response_messages, active_tool_calls)       # 更新工具结果内容

                # LLM输出
                elif isinstance(chunk, AIMessageChunk) and chunk.content and node not in SILENT_NODES:
                    self._handle_llm_token(chunk, node, response_messages)

                yield response_messages

        except Exception as e:
            yield f"❌ Error: {str(e)}"

    def clear_session(self):
        self.rag_system.reset_thread()                  # 重置会话
        self.rag_system.observability.flush()