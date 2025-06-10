#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_mcp_server.py ── 連線到已啟動之 FastMCP HTTP 伺服器
"""

import asyncio
from fastmcp import Client

# --- TextContent 兼容匯入 -----------------------------------------
try:
    # fastmcp ≥ 2.2
    from fastmcp.shared.content import TextContent
except ImportError:
    # 舊版 mcp / IDE 快取
    try:
        from mcp.shared.content import TextContent
    except ImportError:           # 最差情況：只做 duck-typing
        class _Dummy:             # type: ignore
            pass
        TextContent = _Dummy      # pylint: disable=invalid-name
# ------------------------------------------------------------------

HOST, PORT = "127.0.0.1", 8765
client = Client(f"http://{HOST}:{PORT}/mcp")         # ← URL 必帶 /mcp

def _to_str(obj) -> str:
    """把 TextContent 或帶 .text 的物件轉成純字串"""
    if hasattr(obj, "text"):
        return str(obj.text)
    return str(obj)

async def main():
    async with client:
        await client.ping()

        # 1) RAG
        rag = await client.call_tool("MCP_tool1",
                                     {"query": "quick sort algorithm"})
        print("\n=== RAG ===")
        for seg in rag:
            print("•", _to_str(seg)[:120])

        # 2) 產生程式碼
        code = _to_str(await client.call_tool(
            "MCP_tool2",
            {"query": "implement quick sort and print result"}))
        print("\n=== Code (首 300 字) ===\n", code[:300], "…")

        # 3) 執行
        log = _to_str(await client.call_tool("MCP_tool3", {}))
        print("\n=== Log (tail) ===\n", log[-300:])

        # 4) 優化
        new_code = _to_str(await client.call_tool("MCP_tool4", {}))
        print("\n=== Improved (首 300 字) ===\n", new_code[:300], "…")

        # 5) 加註解
        commented = _to_str(await client.call_tool("MCP_tool5", {}))
        print("\n=== Commented (首 300 字) ===\n", commented[:300], "…")

if __name__ == "__main__":
    asyncio.run(main())
