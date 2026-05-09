from __future__ import annotations

import ast
import json
import os
from typing import Any, Dict, List, Tuple
from urllib import request

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from backend.enterprise.config import MIN_CONFIDENCE
from backend.enterprise.ops import record_query
from backend.enterprise.retrieval import HybridRetriever
from backend.enterprise.types import ReActStep, UserContext

load_dotenv()


INTENT_RULES = {
    "fact_query": ["是什么", "是什么", "定义", "多少", "谁", "when", "what", "定义"],
    "process": ["流程", "步骤", "怎么", "如何", "审批", "onboarding", "操作"],
    "comparison": ["对比", "区别", "差异", "优缺点", "compare", "vs"],
    "troubleshooting": ["故障", "报错", "失败", "排查", "error", "异常"],
}


class ReActEnterpriseAssistant:
    def __init__(self) -> None:
        self.model = init_chat_model("ollama:qwen3:1.7b", temperature=0)
        self.retriever = HybridRetriever()

    @staticmethod
    def detect_intent(query: str) -> str:
        text = query.lower()
        for intent, words in INTENT_RULES.items():
            if any(w.lower() in text for w in words):
                return intent
        return "fact_query"

    @staticmethod
    def _calculator(expression: str) -> str:
        allowed_nodes = {
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.Mod,
            ast.USub,
            ast.UAdd,
            ast.Constant,
            ast.Load,
            ast.FloorDiv,
        }
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                raise ValueError("Unsupported expression")
        return str(eval(compile(tree, filename="<calc>", mode="eval"), {"__builtins__": {}}, {}))

    @staticmethod
    def _web_search(query: str) -> str:
        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            return "Web search unavailable: TAVILY_API_KEY is not set."

        payload = json.dumps({"api_key": api_key, "query": query, "max_results": 3}).encode("utf-8")
        req = request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = data.get("results", [])
                lines = []
                for r in results:
                    lines.append(f"- {r.get('title', 'Untitled')} | {r.get('url', '')}\\n  {r.get('content', '')[:180]}")
                return "\n".join(lines) if lines else "No web results."
        except Exception as e:
            return f"Web search failed: {e}"

    def _knowledge_search(self, query: str, user: UserContext) -> Tuple[str, List[Dict[str, Any]], float]:
        result = self.retriever.retrieve(query, user_context=user)
        lines = []
        refs = []
        for c in result.chunks:
            src = c.metadata.get("doc_name") or c.metadata.get("source") or "Unknown"
            section = c.metadata.get("chunk_id", "")
            lines.append(f"[{src}#{section}] {c.page_content[:300]}")
            refs.append({"source": src, "section": section, "metadata": c.metadata, "content": c.page_content})
        return "\n\n".join(lines), refs, result.confidence

    def _fact_check(self, claim: str, user: UserContext) -> str:
        result = self.retriever.retrieve(claim, user_context=user)
        if not result.chunks:
            return "No evidence found in internal knowledge base."
        top = result.chunks[0]
        src = top.metadata.get("doc_name") or top.metadata.get("source") or "Unknown"
        return f"Top supporting evidence from [{src}#{top.metadata.get('chunk_id', '')}] with score={top.rerank_score:.3f}."

    def _select_action(self, query: str, intent: str, step_index: int, observations: List[ReActStep]) -> Dict[str, str]:
        history = "\n".join(
            [f"Step {i+1}: action={s.action}; observation={s.observation[:200]}" for i, s in enumerate(observations)]
        )
        prompt = f"""
You are controlling tools in a ReAct loop.
Return strict JSON only with keys: thought, action, action_input.
Allowed actions: knowledge_search, web_search, calculator, fact_check, finish.
Intent: {intent}
User query: {query}
Step index: {step_index}
History: {history}
Policy:
- First step should usually call knowledge_search.
- For process/troubleshooting, you may call web_search if internal evidence is weak.
- Use calculator only for explicit math.
- Use fact_check before finish if confidence is low.
- If enough evidence exists, action=finish and action_input is the final answer in Chinese.
"""
        raw = self.model.invoke(prompt).content
        try:
            data = json.loads(raw)
            return {
                "thought": str(data.get("thought", "")),
                "action": str(data.get("action", "finish")),
                "action_input": str(data.get("action_input", "")),
            }
        except Exception:
            return {
                "thought": "无法解析模型动作，直接基于已有信息回答。",
                "action": "finish",
                "action_input": "信息不足，无法完成可靠回答。",
            }

    def answer(self, query: str, user: UserContext, max_steps: int = 4) -> Dict[str, Any]:
        intent = self.detect_intent(query)
        steps: List[ReActStep] = []
        refs: List[Dict[str, Any]] = []
        confidence = 0.0
        final_answer = ""

        for i in range(max_steps):
            action_spec = self._select_action(query, intent, i + 1, steps)
            action = action_spec["action"]
            action_input = action_spec["action_input"] or query

            if action == "knowledge_search":
                obs, new_refs, confidence = self._knowledge_search(action_input, user)
                refs = new_refs or refs
            elif action == "web_search":
                obs = self._web_search(action_input)
            elif action == "calculator":
                try:
                    obs = self._calculator(action_input)
                except Exception as e:
                    obs = f"Calculator failed: {e}"
            elif action == "fact_check":
                obs = self._fact_check(action_input, user)
            elif action == "finish":
                final_answer = action_input
                obs = "Answer finalized"
            else:
                obs = f"Unknown action: {action}"

            steps.append(
                ReActStep(
                    thought=action_spec["thought"],
                    action=action,
                    action_input=action_input,
                    observation=obs,
                )
            )

            if action == "finish":
                break

        if not final_answer:
            evidence = "\n\n".join([f"- {r['source']}#{r['section']}" for r in refs[:5]])
            synthesis_prompt = f"""
请基于下列证据回答用户问题，并严格使用中文。
用户问题: {query}
意图类型: {intent}
证据:\n{evidence}
要求:
1) 给出简洁、专业回答。
2) 每个核心结论必须标注来源，格式为 [文档名#章节]。
3) 如果证据不足，明确说明“信息不足”。
"""
            final_answer = str(self.model.invoke(synthesis_prompt).content)

        if confidence < MIN_CONFIDENCE:
            final_answer = f"{final_answer}\n\n信息不足：当前检索证据置信度偏低，建议补充更具体的问题或上传相关文档。"

        record_query(query=query, confidence=confidence, intent=intent)

        citations = []
        for r in refs:
            citations.append(
                {
                    "source": r["source"],
                    "section": r["section"],
                    "department": r["metadata"].get("department", "unknown"),
                    "version": r["metadata"].get("version", "unknown"),
                    "effective_time": r["metadata"].get("effective_time", "unknown"),
                }
            )

        return {
            "intent": intent,
            "answer": final_answer,
            "confidence": confidence,
            "steps": [s.__dict__ for s in steps],
            "citations": citations,
        }
