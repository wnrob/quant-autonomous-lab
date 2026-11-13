"""RD-Agent adapter - LLM BACKEND ONLY. Runs RD-Agent's LiteLLM client against local Ollama inside sandbox_run.sh.
Deliberately NOT exposed: RD-Agent scenarios/loops (they execute LLM-generated code via docker/conda on the host; Docker is absent here).
Contract: chat(prompt, model=...) -> str."""
from __future__ import annotations
import json, os, signal, subprocess
from pathlib import Path
from engine.errors import AdapterUnavailable
ROOT = Path(__file__).resolve().parents[1]

def chat(prompt: str, model: str = "qwen2.5:14b", timeout: int = 300) -> str:
    py = ROOT / "envs/agents/bin/python"
    if not py.exists(): raise AdapterUnavailable("envs/agents not built")
    sbx = ROOT / "artifacts/rdagent_sbx"; sbx.mkdir(parents=True, exist_ok=True)
    env = [f"LITELLM_CHAT_MODEL=ollama_chat/{model}", "OLLAMA_API_BASE=http://127.0.0.1:11434", "LITELLM_CHAT_STREAM=false",
           "LITELLM_ENABLE_RESPONSE_SCHEMA=false", "USE_CHAT_CACHE=False", "USE_EMBEDDING_CACHE=False"]
    cmd = [str(ROOT / "scripts/sandbox_run.sh"), ".", *env, "--", str(py), "-I", "-W", "ignore", str(ROOT / "adapters/workers/rdagent_llm_worker.py"), json.dumps({"prompt": prompt})]
    p = subprocess.Popen(cmd, cwd=sbx, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)   # own process group
    try: out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL); p.communicate()                  # kill the whole tree, not just the wrapper
        raise AdapterUnavailable(f"rdagent llm backend timed out after {timeout}s (process group killed)")
    class r: returncode, stdout, stderr = p.returncode, out, err
    if r.returncode != 0: raise AdapterUnavailable(f"rdagent llm backend failed: {r.stderr[-400:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])["reply"]
