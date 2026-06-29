# evaluate_docker_rag.py
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import json
import sys
import time  # 新增：用于性能计时
from pathlib import Path
from datasets import Dataset
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from ragas.llms import llm_factory
from dotenv import load_dotenv

load_dotenv()

# ================== Ragas 配置 ==================
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness
from ragas.llms import LangchainLLMWrapper

from project import config

print("🔧 Configuring Ragas LLM...")
# 配置评估用 LLM（使用 OpenAI SDK 的 Langchain 包装）
ragas_llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com",
    temperature=0,
    max_tokens=8192,  # ✅ 解决 max_tokens 截断问题
)
print(f"✓ Using deepseek-v4-flash for LLM")

# 配置 Ragas 使用的 Embeddings 模型（使用本地 SentenceTransformer）
print("🔧 Configuring Ragas Embeddings...")
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
embeddings_model = HuggingFaceEmbeddings(model_name=config.DENSE_MODEL)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings_model)
print(f"✓ Using {config.DENSE_MODEL} for embeddings")

# 初始化 Ragas 指标（新版 API 需要实例化）
# faithfulness_metric = Faithfulness()
# answer_relevancy_metric = AnswerRelevancy(strictness=1)                                          # 反向工程：根据答案反向生成 N (strictness设置了N的值)个人工问题，将这些生成的问题与原始用户问题进行embedding并计算余弦相似度，计算平均得分
# context_precision_metric = ContextPrecision()
context_recall_metric = ContextRecall()
answer_correctness_metric = AnswerCorrectness()

# ================== 加载数据集 ==================
dataset_path = Path(__file__).parent.parent.parent / "dataset" / "temp.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ================== 配置区 ==================
DATASET_PATH = str(dataset_path)

# 全局加载一次 RAG 系统
sys.path.insert(0, str(Path(__file__).parent.parent))
from project.core.rag_system import RAGSystem

rag_system = RAGSystem()
rag_system.initialize()

def extract_contexts_from_state(result: dict) -> list:
    """
    从 LangGraph State 中提取检索到的上下文

    策略（按优先级）：
    1. 优先从 agent_answers → contexts 读取（新版 collect_answer 提取的检索内容）
    2. 降级扫描 messages 中的 ToolMessage（兼容旧版/未升级时的调用结果）
    """

    # ① 优先：从 agent_answers 读取
    agent_answers = result.get("agent_answers", [])
    if agent_answers:
        all_ctx = []
        for ans in agent_answers:
            if isinstance(ans, dict) and "contexts" in ans:
                all_ctx.extend(ans["contexts"])
        if all_ctx:
            seen = set()
            unique = []
            for c in all_ctx:
                if c not in seen:
                    unique.append(c)
                    seen.add(c)
            return unique

    # ② 降级：从 messages 扫描 ToolMessage（兼容旧版）
    contexts = []
    seen_contents = set()

    if not isinstance(result, dict) or "messages" not in result:
        return [""]

    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            content = msg.content
            if content and content not in seen_contents:
                if not any(keyword in content for keyword in [
                    "NO_RELEVANT_CHUNKS",
                    "NO_PARENT_DOCUMENTS",
                    "NO_PARENT_DOCUMENT",
                    "RETRIEVAL_ERROR",
                    "PARENT_RETRIEVAL_ERROR"
                ]):
                    contexts.append(content)
                    seen_contents.add(content)

    return contexts if contexts else [""]


def run_rag(question: str, thread_id: str = "eval_thread_001") -> dict:
    """
    调用 Agentic RAG，返回 answer + contexts（用于 Ragas 评估）
    """
    answer = ""
    contexts = [""]
    latency = 0.0

    try:

        start_time = time.time()

        config = {"configurable": {"thread_id": thread_id}}
        input_state = {"messages": [("human", question)]}

        result = rag_system.agent_graph.invoke(input_state, config=config)

        end_time = time.time()  # 新增：结束计时
        latency = end_time - start_time  # 新增：计算耗时（秒）

        # 提取最终答案（最后一个 AI 消息）
        from langchain_core.messages import AIMessage
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                answer = msg.content
                break

        if not answer:
            answer = result["messages"][-1].content if "messages" in result else str(result)

        # 提取检索上下文
        contexts = extract_contexts_from_state(result)

    except Exception as e:
        print(f"❌ RAG 调用错误: {e}")
        answer = f"Error: {str(e)}"
        contexts = [""]
        latency = 0.0  # 错误情况下延迟为 0

    return {
        "answer": answer,
        "contexts": contexts,
        "latency": latency  # 新增：返回延迟
    }

print(f"🚀 开始评估 Docker RAG 系统 - 共 {len(data['samples'])} 个问题")

questions = []
ground_truths = []
answers = []
contexts_list = []
latencies = []  # 新增：存储延迟数据

for i, sample in enumerate(data["samples"]):
    print(f"[{i + 1}/{len(data['samples'])}] {sample['question'][:70]}...")

    output = run_rag(sample["question"], thread_id=f"eval_{i}")

    questions.append(sample["question"])
    ground_truths.append(sample["ground_truth"])
    answers.append(output["answer"])
    contexts_list.append(output["contexts"])
    latencies.append(output["latency"])

# ================== 执行 Ragas 评估 ==================
eval_dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts_list,
    "ground_truth": ground_truths
})

print("📊 正在计算评估指标（可能需要几分钟）...")
result = evaluate(
    eval_dataset,
    # metrics=[faithfulness_metric, answer_relevancy_metric, context_precision_metric, context_recall_metric, answer_correctness_metric],
    metrics=[context_recall_metric, answer_correctness_metric],
    llm=ragas_llm,  # 显式传入 LLM
    embeddings=ragas_embeddings
)

print("\n🎉 === 评估结果 ===")
print(result)

# 保存详细结果
df = result.to_pandas()
df['latency_seconds'] = latencies
# 新增：计算统计信息
import numpy as np
avg_latency = np.mean(latencies)
median_latency = np.median(latencies)
min_latency = np.min(latencies)
max_latency = np.max(latencies)
p95_latency = np.percentile(latencies, 95)

print(f"\n⏱️  === 性能指标（Latency）===")
print(f"平均响应时间: {avg_latency:.2f} 秒")
print(f"中位数响应时间: {median_latency:.2f} 秒")
print(f"最小响应时间: {min_latency:.2f} 秒")
print(f"最大响应时间: {max_latency:.2f} 秒")
print(f"P95 响应时间: {p95_latency:.2f} 秒")


output_csv = Path(__file__).parent / "evalDir" / "docker" / "final_test.csv"
output_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_csv, index=False)
print(f"✅ 详细结果已保存至 {output_csv}")
