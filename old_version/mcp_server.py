#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mcp_server.py ── LOCAL_LLM_CODER FastMCP 伺服器（HTTP 版）
author : Weng Chia-Chun
"""

import os, re, json, argparse, asyncio, logging, mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from fastmcp import FastMCP           # fastmcp ≥ 2.2

# ────── 嘗試載入可選解析套件 ──────
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

# ────── 路徑 & 參數 ──────
BASE_DIR  = Path(__file__).parent
UTILS_DIR = BASE_DIR / "utils"
RAG_PATH  = BASE_DIR / "rag.json"
TEMP_PY   = BASE_DIR / "temp.py"
TEMP_LOG  = BASE_DIR / "temp.log"

LLAMA_PATH  = BASE_DIR / "models" / "generator" / "codellama-7b-instruct.Q4_0.gguf"
RAW_EMB_DIR = BASE_DIR / "models" / "retriever" / "models--sentence-transformers--all-MiniLM-L6-v2"

LLAMA_CTX, LLAMA_GPU_LAYERS = 4096, 40
LLAMA_TEMP, LLAMA_TOP_P, LLAMA_MAX_TOKENS = 0.2, 0.95, 1024
CONDA_ENV_NAME = "autocoder"

# ────── 工具函式 ──────
def _find_st_dir(base: Path) -> str:
    for root, _, files in os.walk(base):
        if "config.json" in files:
            return str(Path(root))
    raise FileNotFoundError("config.json not found")

def _chunk(txt: str, mx: int = 400) -> List[str]:
    lines, buf, out = txt.splitlines(), [], []
    for ln in lines:
        if not ln.strip():
            if buf: out.append("\n".join(buf)); buf=[]
            continue
        buf.append(ln)
        if sum(len(x) for x in buf) >= mx:
            out.append("\n".join(buf)); buf=[]
    if buf: out.append("\n".join(buf))
    return out

# ────── 檔案讀取助手 ──────
def _read_file(fp: Path) -> str:
    """
    依副檔名選擇合適的文字擷取方式。
    若缺少對應套件或解析失敗，回傳空字串。
    """
    ext = fp.suffix.lower()
    try:
        if ext in {".txt", ".md", ".py", ".c", ".cpp", ".h"}:
            return fp.read_text("utf-8", errors="ignore")

        if ext == ".pdf" and pdfplumber:
            with pdfplumber.open(fp) as pdf:
                return "\n".join((page.extract_text() or "") for page in pdf.pages)

        if ext == ".docx" and DocxDocument:
            doc = DocxDocument(fp)
            return "\n".join(p.text for p in doc.paragraphs)

    except Exception:
        pass   # 任何解析失敗都視同空內容
    return ""

# ────── 載入模型 ──────
print("▸ Loading embedding model …")
embedder = SentenceTransformer(_find_st_dir(RAW_EMB_DIR))

print("▸ Loading CodeLlama …")
llm = Llama(model_path=str(LLAMA_PATH),
            n_ctx=LLAMA_CTX, n_gpu_layers=LLAMA_GPU_LAYERS,
            logits_all=False, verbose=False)

# ────── RAG 索引 ──────
@dataclass
class RAG: 
    chunks: List[str]
    vecs:   np.ndarray

def _build_rag() -> RAG:
    chunks: List[str] = []

    for fp in UTILS_DIR.rglob("*.*"):
        txt = _read_file(fp)
        if not txt.strip():
            continue

        for seg in _chunk(txt):
            chunks.append(f"[{fp.name}] {seg.strip()}")

    if not chunks:
        raise RuntimeError("No valid chunks were indexed from utils directory.")

    vecs = embedder.encode(
        chunks, convert_to_numpy=True, normalize_embeddings=True
    )
    return RAG(chunks, vecs)

RAG_DB = _build_rag()
print(f"▸ Indexed {len(RAG_DB.chunks)} segments")

def rag_search(q: str, k: int = 3) -> List[Tuple[str, float]]:
    qv = embedder.encode([q], convert_to_numpy=True,
                         normalize_embeddings=True)[0]
    scr = np.dot(RAG_DB.vecs, qv)
    idx = np.argsort(scr)[::-1][:k]
    return [(RAG_DB.chunks[i], float(scr[i])) for i in idx]

def llamagen(prompt: str) -> str:
    """呼叫 CodeLlama 生成（已去除 <s> 前綴）。"""
    template = (
        "[INST] You are a precise and pragmatic engineer.\n\n"
        f"{prompt.strip()}\n[/INST]"
    )
    out = llm(template, temperature=LLAMA_TEMP, top_p=LLAMA_TOP_P,
              max_tokens=LLAMA_MAX_TOKENS, stop=["</s>"])
    return out["choices"][0]["text"].strip()

# ────── FastMCP Tools ──────
server = FastMCP("LOCAL_LLM_CODER Server")

@server.tool("MCP_tool1")
async def tool1(query: str) -> List[str]:
    loop = asyncio.get_running_loop()
    rs = await loop.run_in_executor(None, rag_search, query, 3)
    rag = [{"rank": i + 1, "score": s, "text": t} for i, (t, s) in enumerate(rs)]
    RAG_PATH.write_text(json.dumps(rag, ensure_ascii=False, indent=2), "utf-8")
    return [d["text"] for d in rag]

@server.tool("MCP_tool2")
async def tool2(query: str) -> str:
    rag = json.loads(RAG_PATH.read_text("utf-8")) if RAG_PATH.exists() else []
    rag_txt = "\n---\n".join(d["text"] for d in rag) or "（無）"
    prm = (
        f"### User requirement:\n{query}\n\n"
        f"### Retrieved context (RAG):\n{rag_txt}\n\n"
        "If the RAG context is irrelevant or unhelpful, **ignore it** and "
        "focus strictly on the user requirement. "
        "Return runnable **Python 3** code only."
    )
    loop = asyncio.get_running_loop()
    raw = await loop.run_in_executor(None, llamagen, prm)
    code = re.findall(r"```(?:python)?\n(.+?)```", raw, re.S)
    final = code[0] if code else raw
    TEMP_PY.write_text(final, "utf-8")
    return final

@server.tool("MCP_tool3")
async def tool3() -> str:
    if not TEMP_PY.exists():
        raise FileNotFoundError("temp.py 未產生")
    proc = await asyncio.create_subprocess_exec(
        "conda", "run", "-n", CONDA_ENV_NAME, "python", str(TEMP_PY),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    out, _ = await proc.communicate()
    log = out.decode("utf-8", errors="ignore")
    if not log.strip():
        log = "(no output)"
    TEMP_LOG.write_text(log, "utf-8")
    return log[-500:]

@server.tool("MCP_tool4")
async def tool4() -> str:
    code = TEMP_PY.read_text("utf-8")
    log  = TEMP_LOG.read_text("utf-8") if TEMP_LOG.exists() else "(no output)"
    rag  = json.loads(RAG_PATH.read_text("utf-8")) if RAG_PATH.exists() else []
    prm = (
        f"### Current code\n```python\n{code}\n```\n\n"
        f"### Execution log\n{log[-2000:]}\n\n"
        f"### RAG\n" + "\n---\n".join(d["text"] for d in rag) + "\n\n"
        "Fix bugs or improve performance; return **complete code only**."
    )
    loop = asyncio.get_running_loop()
    raw = await loop.run_in_executor(None, llamagen, prm)
    code = re.findall(r"```(?:python)?\n(.+?)```", raw, re.S)
    final = code[0] if code else raw
    TEMP_PY.write_text(final, "utf-8")
    return final

@server.tool("MCP_tool5")
async def tool5() -> str:
    code = TEMP_PY.read_text("utf-8")
    prm = (
        "Add clear **English** comments to the following Python 3 code "
        "(keep logic unchanged). Return the commented code only.\n"
        f"```python\n{code}\n```"
    )
    loop = asyncio.get_running_loop()
    raw = await loop.run_in_executor(None, llamagen, prm)
    code = re.findall(r"```(?:python)?\n(.+?)```", raw, re.S)
    final = code[0] if code else raw
    TEMP_PY.write_text(final, "utf-8")
    return final

# ────── 入口 ──────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-H", "--host", default="127.0.0.1")
    ap.add_argument("-p", "--port", type=int, default=8765)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    server.run(
        transport="streamable-http",
        host=args.host,
        port=args.port,
    )
