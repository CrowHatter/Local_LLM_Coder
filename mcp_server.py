from fastmcp import FastMCP
import ast, pathlib

mcp = FastMCP("local_llm_coder")

# ---------- Tool #1 ---------------------------------------------------
@mcp.tool()
def test_code_snippet(code: str, test_expr: str) -> str:
    """執行 `code`，回傳 `eval(test_expr)` 結果。"""
    ns: dict = {}
    exec(code, {}, ns)
    return str(eval(test_expr, {}, ns))

# ---------- Tool #2 〈改版：用 ast.parse〉 ------------------------------
@mcp.tool()
def analyze_code(code: str) -> str:
    """
    嘗試以 ast.parse 解析程式碼：
    - 成功 ⇒ 回傳 'Syntax OK'
    - SyntaxError ⇒ 回傳錯誤行號與訊息
    省掉 flake8，避免 stdio deadlock。
    """
    try:
        ast.parse(code)
        return "Syntax OK"
    except SyntaxError as e:
        return f"SyntaxError: {e.lineno}:{e.offset} {e.msg}"

# ---------- Tool #3 ---------------------------------------------------
@mcp.tool()
def parse_project_structure(root_path: str) -> str:
    """遞迴列出 root_path 內所有 .py（相對路徑）。"""
    return "\n".join(
        sorted(
            p.relative_to(root_path).as_posix()
            for p in pathlib.Path(root_path).rglob("*.py")
        )
    )

if __name__ == "__main__":
    mcp.run()        # 預設 transport=stdio
