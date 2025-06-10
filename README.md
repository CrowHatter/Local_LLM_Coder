# 專題：Local LLM Coder – 5 Agents + 3 MCP Tools

## 一、專題概述
- **目標**：建立一個由 Client 端 5 個 Agent（Retriever → Planner → Coder ↔ Executor ↔ Debugger）和 Server 端 3 個 MCP Tools（RAG_Search / WriteTemp / ExecTemp）組成的自動化程式碼生成與執行流程。  
- **核心流程**：
  1. **RetrieverAgent**：從 `utils/` 資料夾讀取並向量化文件，回傳 top-k chunks  
  2. **PlannerAgent**：結合使用者需求與 RAG context 草擬「計畫（PLAN）」  
  3. **CoderAgent**：嚴格依照 PLAN 生成 Python3 代碼（main()、無參數）  
  4. **ExecutorAgent**：將代碼寫入 `temp.py` → 執行 → 收集 `temp.log` → 用 LLM judge 是否符合 PLAN  
  5. **DebuggerAgent**：根據 PLAN＋code＋log＋reason 修正 PLAN，迴圈直到成功或 5 輪結束

## 二、環境依賴與先行準備
1. **Python 3.12+ & 虛擬環境**  
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install --upgrade pip
   ```
2. **Conda `autocoder` 環境**（執行 `temp.py` 時必須）  
   ```bash
   conda create -n autocoder python=3.12 -y
   conda activate autocoder
   pip install fastmcp sentence-transformers pdfplumber python-docx httpx
   ```
3. **下載並放置模型**  
   - 前往：https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF  
   - 手動下載 `codellama-7b-instruct.Q4_0.gguf`，放到：  
     ```
     models/generator/codellama-7b-instruct.Q4_0.gguf
     ```
4. **CUDA 12.6 + GPU 加速**  
   - 安裝 CUDA Toolkit v12.6，不需移除  
   - 安裝 VS 2022 Build Tools（含 C++ / CMake）  
   - 編譯並安裝 GPU 版 `llama-cpp-python`：
     ```PowerShell
      $env:CUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
      $env:CMAKE_GENERATOR_PLATFORM="x64"
      $env:FORCE_CMAKE="1"
      $env:CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89"
      pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
     ```
   - 測試 GPU：
     ```python
     from llama_cpp import Llama
     llm = Llama(
         "models/generator/codellama-7b-instruct.Q4_0.gguf",
         n_ctx=8192, n_gpu_layers=40, verbose=True
     )
     ```
5. **資料與權限**  
   - `models/retriever/...` (all-MiniLM-L6-v2) 目錄可讀  
   - `utils/` 放入 *.txt, .md, .py, .pdf, .docx…  
   - `log.json`, `temp.py`, `temp.log` 需可寫

## 三、常見問題與對策
- **Judge 解析失敗**：ExecutorAgent 現在容錯，若非 JSON 但無 Traceback，仍視為通過  
- **死迴圈**：DebuggerAgent 若 reason 重複，標記 `(no-judge)`，避免反覆  
- **Token 不夠**：已升至 `n_ctx=8192`，Coder `max_tokens=3072`  
- **GPU 仍 CPU**：檢查 `CUDA\v12.6\bin` 在 PATH 前面，DLL 是否可被 `find_library`

## 四、檔案清單
- `MCP.py`：Server 端 3 大工具  
- `Client.py`：Client 端 5 Agent + 完整 logging  
- `gpu_test.py`：GPU 驗證程式  
- `utils/`：RAG 文件資料夾  
- `log.json`：執行記錄

---
