#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Agent Web 服务器 - FastAPI 后端
提供 HTTP API 供前端调用
"""

import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "http://127.0.0.1:2024"   # ← 只在这里写死！
os.environ["LANGCHAIN_PROJECT"] = "Vision-Agent-Dev"

import uuid
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn

from agents import chat_agent, _image_context


app = FastAPI(title="Vision Agent API", version="1.0.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 会话管理（简单的内存存储）
sessions = {}


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str
    session_id: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def root():
    """返回前端页面"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/chat")
async def chat(
    message: str = Form(...),
    session_id: str = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    处理聊天请求

    参数:
    - message: 用户消息
    - session_id: 会话ID（可选，用于多轮对话）
    - image: 上传的图片（可选）
    """
    try:
        # 准备输入
        input_data = {"input": message}

        # 处理图片
        if image:
            image_bytes = await image.read()
            input_data["image_bytes"] = image_bytes
            input_data["image_filename"] = image.filename or f"upload_{uuid.uuid4().hex[:8]}.jpg"

        # 生成会话ID（如果没有）
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex}"

        # 配置（使用会话ID作为thread_id）
        config = {
            "configurable": {"thread_id": session_id},
            "model_type": "text"
        }

        # 清除上次的检测结果，避免返回旧图片
        _image_context["visualization_b64"] = None
        _image_context["detection_results"] = None

        # 调用 agent
        result = chat_agent.invoke(input_data, config)
        response_text = result.get("output", "抱歉，我没有生成回复。")

        # 获取检测后的可视化图片（如果有）
        visualization_image = _image_context.get("visualization_b64")

        return JSONResponse({
            "success": True,
            "response": response_text,
            "session_id": session_id,
            "visualization": visualization_image  # 返回带框图片
        })

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ 聊天错误:\n{error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reset")
async def reset_session(session_id: str = Form(...)):
    """重置会话"""
    if session_id in sessions:
        del sessions[session_id]

    return JSONResponse({
        "success": True,
        "message": "会话已重置"
    })


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "qwen3-vision-agent"}


if __name__ == "__main__":
    print("🚀 启动 Vision Agent Web 服务器...")
    print("📍 访问地址: http://localhost:7860")
    print("📚 API 文档: http://localhost:7860/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info"
    )
