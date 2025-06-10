#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mcp.py ── Minimal FastMCP server · 3 tools
"""

import os, json, argparse, asyncio, logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from fastmcp import FastMCP

# ────── optional parsers ──────
try: import pdfplumber
except ImportError: pdfplumber = None
try: from docx import Document as DocxDocument
except ImportError: DocxDocument = None

# ────── paths ──────
BASE_DIR  = Path(__file__).parent
UTILS_DIR = BASE_DIR / "utils"
RAG_PATH  = BASE_DIR / "rag.json"
TEMP_PY   = BASE_DIR / "temp.py"
TEMP_LOG  = BASE_DIR / "temp.log"
EMB_DIR   = (BASE_DIR / "models" / "retriever"
             / "models--sentence-transformers--all-MiniLM-L6-v2")

# ────── helpers ──────
def _find_st_dir(base: Path) -> str:
    for root, _, files in os.walk(base):
        if "config.json" in files:
            return str(Path(root))
    raise FileNotFoundError

def _chunk(txt: str, mx: int = 400) -> List[str]:
    lines, buf, out = txt.splitlines(), [], []
    for ln in lines:
        if not ln.strip():
            if buf: out.append("\n".join(buf)); buf = []
            continue
        buf.append(ln)
        if sum(len(x) for x in buf) >= mx:
            out.append("\n".join(buf)); buf = []
    if buf: out.append("\n".join(buf))
    return out

def _read_file(fp: Path) -> str:
    ext = fp.suffix.lower()
    try:
        if ext in {".txt", ".md", ".py", ".c", ".cpp", ".h"}:
            return fp.read_text("utf-8", errors="ignore")
        if ext == ".pdf" and pdfplumber:
            with pdfplumber.open(fp) as pdf:
                return "\n".join((p.extract_text() or "") for p in pdf.pages)
        if ext == ".docx" and DocxDocument:
            doc = DocxDocument(fp); return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        pass
    return ""

# ────── build RAG index ──────
embedder = SentenceTransformer(_find_st_dir(EMB_DIR))

@dataclass
class RAG: chunks: List[str]; vecs: np.ndarray

def _build_rag() -> RAG:
    chunks: List[str] = []
    for fp in UTILS_DIR.rglob("*.*"):
        txt = _read_file(fp)
        if not txt.strip(): continue
        for seg in _chunk(txt):
            chunks.append(f"[{fp.name}] {seg.strip()}")
    vecs = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    return RAG(chunks, vecs)

RAG_DB = _build_rag()
print(f"▸ Indexed {len(RAG_DB.chunks)} segments")

def _rag_search(query: str, k: int = 3) -> List[str]:
    qv  = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scr = np.dot(RAG_DB.vecs, qv)
    idx = np.argsort(scr)[::-1][:k]
    return [RAG_DB.chunks[i] for i in idx]

# ────── FastMCP server ──────
server = FastMCP("Minimal_RAG_Server")

@server.tool("RAG_Search")
async def rag_search_tool(query: str) -> List[str]:
    chunks = await asyncio.get_running_loop().run_in_executor(None, _rag_search, query)
    RAG_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), "utf-8")
    return chunks  # list[str]

@server.tool("WriteTemp")
async def write_temp_tool(code: str) -> dict:
    TEMP_PY.write_text(code, "utf-8")
    return {"ok": True}

@server.tool("ExecTemp")
async def exec_temp_tool(payload: dict | None = None) -> dict:
    if not TEMP_PY.exists():
        return {"success": False, "log": "temp.py not found"}
    proc = await asyncio.create_subprocess_exec(
        "conda", "run", "-n", "autocoder", "python", str(TEMP_PY),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    out, _ = await proc.communicate()
    log = out.decode("utf-8", errors="ignore")
    TEMP_LOG.write_text(log, "utf-8")
    return {"success": proc.returncode == 0, "log": log}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-H", "--host", default="0.0.0.0")
    ap.add_argument("-p", "--port", type=int, default=8765)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    server.run(transport="streamable-http",
               host=args.host, port=args.port)
