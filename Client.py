from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client_orchestrator.py ── 5-Agent workflow  +  extended tokens + full logging
"""

import asyncio, json, re, datetime
from pathlib import Path
from typing import List, Any, Optional, Dict

from fastmcp import Client
from llama_cpp import Llama

# ────── 路徑 & 參數 ──────
BASE_DIR   = Path(__file__).parent
LLAMA_PATH = BASE_DIR / "models" / "generator" / "codellama-7b-instruct.Q4_0.gguf"
LOG_PATH   = BASE_DIR / "log.json"
MCP_BASE   = "http://127.0.0.1:8765/mcp"
MAX_ROUND  = 5
client     = Client(MCP_BASE)

# ────── TextContent 轉字串 ──────
try:
    from fastmcp.shared.content import TextContent
except ImportError:
    class TextContent: pass                                        # type: ignore


def _to_str(obj: Any) -> str:
    return str(obj.text) if hasattr(obj, "text") else str(obj)


# ────── Llama 單例 ──────
class LlamaWrapper:
    _inst: Optional["LlamaWrapper"] = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
            cls._inst.llm = Llama(
                model_path=str(LLAMA_PATH),
                n_ctx=8192,          # ↑ token 上限拉高
                n_gpu_layers=40,     # VRAM 夠可 -1
                verbose=False
            )
        return cls._inst

    def chat(self, system: str, prompt: str,
             temperature: float = 0.15,
             top_p: float = 0.9,
             max_tokens: int = 2048) -> str:
        msg = f"[INST] <SYS>\n{system}\n</SYS>\n\n{prompt.strip()} [/INST]"
        out = self.llm(msg, temperature=temperature,
                       top_p=top_p, max_tokens=max_tokens,
                       stop=["</s>"])
        return out["choices"][0]["text"].strip()


LLM = LlamaWrapper()


async def call_tool(name: str, payload):
    async with client:
        return await client.call_tool(name, payload)


# ────── logging util ──────
def _log(agent: str, info: Dict[str, Any]) -> None:
    rec = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
           "agent": agent, **info}
    try:
        data = json.loads(LOG_PATH.read_text("utf-8")) if LOG_PATH.exists() else []
    except Exception:
        data = []
    data.append(rec)
    LOG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


# ────── Agents ──────
class RetrieverAgent:
    async def run(self, query: str) -> List[str]:
        res = await call_tool("RAG_Search", {"query": query})
        chunks = [_to_str(x) for x in (res if isinstance(res, list) else res.get("chunks", []))]
        _log("Retriever", {"chunks": len(chunks)})
        return chunks


class PlannerAgent:
    SYSTEM = ("You are a senior software architect. "
              "Write a concise implementation PLAN in English. No code.")

    async def run(self, user_prompt: str, context: List[str]) -> str:
        ctx = "\n---\n".join(context) or "(none)"
        prompt = (f"### USER PROMPT (keep this line intact!)\n{user_prompt}\n\n"
                  f"### RELEVANT CONTEXT\n{ctx}")
        plan = LLM.chat(self.SYSTEM, prompt, max_tokens=2048)
        _log("Planner", {"preview": plan[:120]})
        return plan


class CoderAgent:
    SYSTEM = ("You are a disciplined Python engineer.\n"
              "- Code must run from main().\n"
              "- No input/argparse.\n"
              "- No blank lines.\n"
              "- Output ONE fenced Python 3 block ONLY.\n"
              "- If you have not enough room, shorten comments, not code.")
    CODE_RE = re.compile(r"```(?:python)?\s*\n([\s\S]+?)```", re.I)

    async def run(self, plan: str) -> str:
        raw = LLM.chat(self.SYSTEM, f"### PLAN\n{plan}", max_tokens=3072)
        m = self.CODE_RE.search(raw)
        code = m.group(1).strip() if m else ""
        _log("Coder", {"code_len": len(code)})
        return code


class ExecutorAgent:
    SYSTEM_JUDGE = ("You are an automatic judge. "
                    "Return JSON {\"ok\":bool, \"reason\":\"...\"}.")

    async def run(self, code: str, plan: str) -> dict:
        if not code:
            _log("Executor", {"error": "no code"})
            return {"success": False, "reason": "no code", "log": "", "code": code}

        await call_tool("WriteTemp", {"code": code})
        res = await call_tool("ExecTemp", {})
        log_full = res["log"] if isinstance(res, dict) else _to_str(res)
        runtime_ok = res.get("success", False) if isinstance(res, dict) else False

        judge_prompt = f"### PLAN\n{plan}\n\n### LOG\n{log_full[-800:]}"
        try:
            judge = json.loads(LLM.chat(self.SYSTEM_JUDGE, judge_prompt, max_tokens=512))
            output_ok, reason = bool(judge.get("ok")), judge.get("reason", "")
        except Exception:
            output_ok, reason = False, "judge parse error"

        success = runtime_ok and output_ok
        _log("Executor", {"runtime_ok": runtime_ok, "output_ok": output_ok, "reason": reason})
        return {"success": success, "code": code, "log": log_full, "reason": reason}


class DebuggerAgent:
    SYSTEM = ("You are a senior debugger. Update ONLY the PLAN (keep the USER PROMPT line).")

    async def run(self, plan: str, code: str, log: str, reason: str) -> str:
        prompt = (f"### CURRENT PLAN\n{plan}\n\n"
                  f"### CODE HEAD\n{code[:400]}\n\n"
                  f"### LOG TAIL\n{log[-400:]}\n\n"
                  f"### ISSUE\n{reason}")
        new_plan = LLM.chat(self.SYSTEM, prompt, max_tokens=2048)
        _log("Debugger", {"preview": new_plan[:120]})
        return new_plan


# ────── Workflow ──────
async def run_workflow(requirement: str):
    LOG_PATH.write_text("[]", "utf-8")            # reset

    ret, pla = RetrieverAgent(), PlannerAgent()
    cod, exe, dbg = CoderAgent(), ExecutorAgent(), DebuggerAgent()

    ctx = await ret.run(requirement)
    plan = await pla.run(requirement, ctx)

    for rnd in range(1, MAX_ROUND + 1):
        _log("Orchestrator", {"round": rnd})

        code = await cod.run(plan)
        exe_res = await exe.run(code, plan)

        if exe_res["success"]:
            Path("final_code.py").write_text(code, "utf-8")
            _log("Orchestrator", {"status": "success"})
            break

        plan = await dbg.run(plan, exe_res["code"], exe_res["log"], exe_res["reason"])
    else:
        _log("Orchestrator", {"status": "max_round_reached"})


# ────── CLI ──────
if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="5-Agent orchestrator w/ extended tokens")
    ap.add_argument("requirement", nargs="*", help="user prompt (blank → input)")
    args = ap.parse_args()

    prompt = " ".join(args.requirement).strip() if args.requirement else input("🔹 prompt: ").strip()
    if not prompt:
        print("❌ empty prompt"); sys.exit(1)

    asyncio.run(run_workflow(prompt))
