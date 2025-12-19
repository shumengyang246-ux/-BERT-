# Soft-Mask-BERT模型以及调用deepseek大模型的API服务
import json
import os
import urllib.request
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference import CSCInference

APP_DIR = os.path.dirname(__file__)
WEB_DIR = os.path.abspath(os.path.join(APP_DIR, "..", "web"))

SOFTMASKED_CKPT = os.getenv(
    "SOFTMASKED_CKPT",
    r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\final_model\ckpt_best.pt",
)

DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-5966a395d4fc41dbba6f1e465e54af8d")

QWEN_API_URL = os.getenv(
    "QWEN_API_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
)
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3-max")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "sk-bba61c1a153142d0aae738b7a06f8a3e")


app = FastAPI(title="Soft-Mask-BERT CSC Demo", version="1.0")

_engine: Optional[CSCInference] = None

# 纠错请求
class CorrectRequest(BaseModel):
    text: str
    model: str = "softmasked"

# 纠错响应
class CorrectResponse(BaseModel):
    model: str
    raw: str
    corrected: str
    edits: List[Dict[str, Any]]

# 获取推理引擎
def get_engine() -> CSCInference:
    global _engine
    if _engine is None:
        _engine = CSCInference(SOFTMASKED_CKPT)
    return _engine

# 把“原文 vs 纠错结果”转成可展示的编辑操作
def build_edits(src: str, tgt: str) -> List[Dict[str, Any]]:
    edits: List[Dict[str, Any]] = []
    matcher = SequenceMatcher(a=src, b=tgt)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            span = min(i2 - i1, j2 - j1)
            for k in range(span):
                edits.append(
                    {
                        "type": "replace",
                        "index": i1 + k,
                        "src_char": src[i1 + k],
                        "pred_char": tgt[j1 + k],
                    }
                )
            for k in range(i1 + span, i2):
                edits.append(
                    {
                        "type": "delete",
                        "index": k,
                        "src_char": src[k],
                        "pred_char": "",
                    }
                )
            for k in range(j1 + span, j2):
                edits.append(
                    {
                        "type": "insert",
                        "index": i1 + span,
                        "src_char": "",
                        "pred_char": tgt[k],
                    }
                )
        elif tag == "delete":
            for k in range(i1, i2):
                edits.append(
                    {
                        "type": "delete",
                        "index": k,
                        "src_char": src[k],
                        "pred_char": "",
                    }
                )
        elif tag == "insert":
            for k in range(j1, j2):
                edits.append(
                    {
                        "type": "insert",
                        "index": i1,
                        "src_char": "",
                        "pred_char": tgt[k],
                    }
                )
    return edits

# 调用DeepSeek API进行纠错
def deepseek_correct(text: str) -> str:
    api_key = DEEPSEEK_API_KEY
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing DeepSeek API key.")

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个中文文本纠错助手，精通简体中文和繁体中文的语法和拼写纠正。"
                    "请一定注意：繁体中文句子中的错别字也需要纠正为正确的繁体中文用词，而不是转换为简体中文；没有错的繁体中文也不要转换为简体中文。"
                    "只返回纠正后的文本，不要解释或额外格式。"
                ),
            },
            {"role": "user", "content": text},
        ],
        "temperature": 0.3,
    }

    req = urllib.request.Request(
        DEEPSEEK_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"DeepSeek request failed: {exc}") from exc

    try:
        data = json.loads(body)
        corrected = data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise HTTPException(status_code=502, detail="DeepSeek response parse failed.") from exc

    return corrected

# 调用通义千问API进行纠错
def qwen_correct(text: str) -> str:
    api_key = QWEN_API_KEY
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing Qwen API key.")
    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个中文文本纠错助手，精通简体中文和繁体中文的语法和拼写纠正。"
                    "请一定注意：繁体中文句子中的错别字也需要纠正为正确的繁体中文用词，而不是转换为简体中文；没有错的繁体中文也不要转换为简体中文。"
                    "只返回纠正后的文本，不要解释或额外格式。"
                ),
            },
            {"role": "user", "content": text},
        ],
        "temperature": 0.3,
    }

    req = urllib.request.Request(
        QWEN_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qwen request failed: {exc}") from exc

    try:
        data = json.loads(body)
        corrected = data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Qwen response parse failed.") from exc

    return corrected


@app.get("/")
def serve_index():
    index_path = os.path.join(WEB_DIR, "index.html")
    return FileResponse(index_path)


@app.post("/api/correct", response_model=CorrectResponse)
def correct(req: CorrectRequest):
    text = req.text or ""
    if not text.strip():
        return CorrectResponse(model=req.model, raw=text, corrected=text, edits=[])

    model_name = (req.model or "softmasked").lower()
    if model_name not in {"softmasked", "deepseek", "qwen"}:
        raise HTTPException(status_code=400, detail="model must be softmasked, deepseek, or qwen")

    if model_name == "softmasked":
        engine = get_engine()
        corrected, edits = engine.predict(text)
        return CorrectResponse(model="softmasked", raw=text, corrected=corrected, edits=edits)

    if model_name == "deepseek":
        corrected = deepseek_correct(text)
        edits = build_edits(text, corrected)
        return CorrectResponse(model="deepseek", raw=text, corrected=corrected, edits=edits)

    corrected = qwen_correct(text)
    edits = build_edits(text, corrected)
    return CorrectResponse(model="qwen", raw=text, corrected=corrected, edits=edits)

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
