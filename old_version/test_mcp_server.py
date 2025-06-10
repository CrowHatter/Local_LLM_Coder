#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_mcp_server.py ── 連線到已啟動之 FastMCP HTTP 伺服器
"""

import asyncio
from fastmcp import Client

# ─── TextContent 相容匯入 ───
try:
    from fastmcp.shared.content import TextContent           # fastmcp ≥ 2.2
except ImportError:
    try:
        from mcp.shared.content import TextContent           # 舊版或 IDE 快取
    except ImportError:                                      # 最後回退
        class TextContent:                                   # type: ignore
            pass
# ────────────────────────────

HOST, PORT = "127.0.0.1", 8765
client = Client(f"http://{HOST}:{PORT}/mcp")                 # URL 必帶 /mcp

def _to_str(obj) -> str:
    """支援 TextContent / 具 .text 屬性 / 其他型別"""
    if hasattr(obj, "text"):
        return str(obj.text)
    return str(obj)

async def main():
    async with client:
        await client.ping()

        # 1) RAG
        rag = await client.call_tool("MCP_tool1",
                                     {"query": "write an recursive example to print 10!"})
        print("\n=== RAG ===")
        for seg in rag:
            print("•", _to_str(seg)[:120])

        # 2) 產生程式碼
        code = _to_str(await client.call_tool(
            "MCP_tool2",
            {"query": "implement write an recursive example and print print 10!"}))
        print("\n=== Code (first 300 chars) ===\n", code[:300], "…")

        # 3) 執行
        log = _to_str(await client.call_tool("MCP_tool3", {}))
        print("\n=== Log (tail) ===\n", log[-300:])

        # 4) 優化
        new_code = _to_str(await client.call_tool("MCP_tool4", {}))
        print("\n=== Improved (first 300 chars) ===\n", new_code[:300], "…")

        # 5) 加註解
        commented = _to_str(await client.call_tool("MCP_tool5", {}))
        print("\n=== Commented (first 300 chars) ===\n", commented[:300], "…")

if __name__ == "__main__":
    asyncio.run(main())
