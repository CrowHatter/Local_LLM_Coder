import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ▶ 修改成你在 Autocoder 環境中的 python.exe
AUTO_PY = r"C:\Users\EricWeng\miniconda3\envs\autocoder\python.exe"

SERVER = StdioServerParameters(
    command=AUTO_PY,
    args=["-u", "mcp_server.py"],   # -u 讓 stderr 立即 flush
    capture_output=True,            # 方便出錯時讀取 server stderr
)

TASK = "Write a recursive factorial(n) function."

async def pipeline(sess: ClientSession) -> None:
    # (1) 產生初版程式
    code = (await sess.call_tool("code_generator", {"task": TASK})).content[0].text

    # (2) 產生測試案例
    tests_raw = (
        await sess.call_tool("test_case_generator", {"task": TASK, "code": code})
    ).content[0].text
    try:
        tests = eval(tests_raw)              # 轉回 List[str]
    except Exception:
        tests = [ln.strip() for ln in tests_raw.splitlines() if ln.strip()]

    # (3) 安裝缺失套件
    await sess.call_tool("dep_installer", {"code": code})

    # (4) 執行測試
    report = (
        await sess.call_tool("code_tester", {"code": code, "tests": tests})
    ).content[0].text

    # (5) 根據測試結果優化程式
    optimized = (
        await sess.call_tool("code_optimizer", {"code": code, "test_report": report})
    ).content[0].text

    # (6) 加入中英文 Docstring
    final_code = (
        await sess.call_tool("docstring_adder", {"task": TASK, "code": optimized})
    ).content[0].text

    print("\n=== Final Code ===\n")
    print(final_code)

async def main() -> None:
    try:
        async with stdio_client(SERVER) as (read, write):
            async with ClientSession(read, write) as sess:
                await sess.initialize()
                await pipeline(sess)
    except Exception:
        # 盡量把 Server stderr 打出來，方便定位問題
        try:
            proc = read.proc                 # type: ignore
            err = proc.stderr.read().decode(errors="ignore")
            print("\n=== Server stderr ===\n", err)
        except Exception:
            print("\n(無法讀取伺服器 stderr)")
        raise

if __name__ == "__main__":
    asyncio.run(main())
