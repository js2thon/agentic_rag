# -*- coding: utf-8 -*-
"""模型服务连通性自检。

用法:
    python scripts/check_llm.py

读取顺序与 app.py 一致: 环境变量 LLM_BASE / LLM_KEY / LLM_MODEL,
或 .streamlit/secrets.toml 里的 llm_base / llm_key / llm_model。
只发一次极小的 chat 请求, 用来确认地址、Key、模型标识是否可用。
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASE = "https://api.openai.com/v1"


def read_secrets() -> dict:
    """极简 TOML 读取, 只认 'key = "value"' 这种单行字符串。"""
    path = os.path.join(ROOT, ".streamlit", "secrets.toml")
    data = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                data[key.strip()] = val.strip().strip('"').strip("'")
    except OSError:
        pass
    return data


def setting(name: str, default: str = "") -> str:
    env = os.getenv(f"LLM_{name.upper()}", "")
    return (env or read_secrets().get(f"llm_{name}", "") or default).strip()


def main() -> int:
    base = setting("base", DEFAULT_BASE).rstrip("/")
    key = setting("key")
    model = setting("model")

    print(f"base  : {base}")
    print(f"model : {model or '(未配置)'}")
    print(f"key   : {'已配置 ' + key[:6] + '…' if key else '(未配置)'}")

    if not (key and model):
        print("\n[FAIL] 缺少 llm_key 或 llm_model: 请写在 .streamlit/secrets.toml "
              "或导出环境变量 LLM_KEY / LLM_MODEL")
        return 1

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 8,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
        method="POST",
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        print(f"\n[FAIL] HTTP {e.code}: {detail}")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"\n[FAIL] {type(e).__name__}: {e}")
        return 1

    cost = time.time() - t0
    text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
    usage = body.get("usage") or {}
    print(f"\n[OK] {cost:.2f}s · tokens={usage.get('total_tokens', '?')} · "
          f"reply={text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
