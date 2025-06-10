import sys, asyncio, tempfile, os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = StdioServerParameters(
    command=sys.executable,
    args=["mcp_server.py"],
)

async def main():
    print("🚀 啟動 MCP Server (stdio)…")
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as sess:
            await sess.initialize()

            # 1) test_code_snippet
            r1 = await sess.call_tool(
                "test_code_snippet",
                {"code": "def add(a,b):\n return a+b", "test_expr": "add(2,3)"}
            )
            print("1) test_code_snippet ➜", r1.content[0].text)

            # 2) analyze_code 〈改用 ast.parse〉
            bad_code = "def f():\nprint('bad indent')\n"
            r2 = await sess.call_tool("analyze_code", {"code": bad_code})
            print("2) analyze_code ➜", r2.content[0].text)

            # 3) parse_project_structure
            with tempfile.TemporaryDirectory() as d:
                os.makedirs(os.path.join(d, "pkg"))
                open(os.path.join(d, "a.py"), "w").close()
                open(os.path.join(d, "pkg", "b.py"), "w").close()
                r3 = await sess.call_tool("parse_project_structure", {"root_path": d})
                print("3) parse_project_structure ➜\n", r3.content[0].text)

if __name__ == "__main__":
    asyncio.run(main())
