"""Smoke-test ONLY RD-Agent's LLM backend (LiteLLM) against the local Ollama model. Does not run any RD-Agent scenario/loop (those execute generated code)."""
import json, sys
from rdagent.oai.llm_utils import APIBackend
a = json.loads(sys.argv[1])
out = APIBackend().build_messages_and_create_chat_completion(user_prompt=a["prompt"], system_prompt="Answer in one short sentence.")
print(json.dumps({"reply": out}))
