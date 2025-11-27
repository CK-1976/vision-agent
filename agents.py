import os
import httpx
import base64
from typing import TypedDict, Annotated, Literal
from langchain.tools import tool
from langchain_qwq import ChatQwen
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from config import settings

# ====================== 状态定义 ======================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  
    image_bytes: bytes | None
    image_filename: str | None
    model_type: Literal["text", "realtime"]

# ====================== 工具 ======================
_image_context = {
    "bytes": None,
    "filename": None,
    "visualization_b64": None,  # 存储带框图片的base64
    "detection_results": None   # 存储坐标
}

@tool
def detect_objects(categories: str) -> dict:
    """
    使用开放词汇目标检测模型检测图片中的物体。支持自然语言描述的任意类别。

    参数:
    - categories: 要检测的类别描述，多个类别用逗号分隔。

    类别提取原则：
    1. **拆解复杂问题**：将用户的复杂问题拆解为独立的检测类别
    2. **保留关键修饰词**：保留颜色、状态、位置等关键属性，但要精简
    3. **一个类别一个概念**：每个类别应该是一个可独立检测的视觉概念
    4. **保持语言一致性**：根据用户的语言习惯提取类别（中文/英文/混合）

    类别提取示例：

    中文输入 → 中文类别：
    - 用户："这里有人吗"
      → categories="人"

    - 用户："图中有几个人和几辆车"
      → categories="人,汽车"

    - 用户："有穿红色衣服的人吗"
      → categories="穿红色衣服的人"

    - 用户："找一下最黄的树"
      → categories="最黄的树"

    - 用户："检测画面中所有戴帽子的人和拿着伞的人"
      → categories="戴帽子的人,拿着伞的人"

    英文输入 → 英文类别：
    - 用户："Are there any people here"
      → categories="person"

    - 用户："Find the person wearing red clothes"
      → categories="person wearing red clothes"

    - 用户："Detect all people, cars, dogs and cats"
      → categories="person,car,dog,cat"

    - 用户："Is there a red car parked on the roadside"
      → categories="red car parked on roadside"

    中英混合输入 → 中英混合类别：
    - 用户："帮我找一下 red car"
      → categories="red car"

    - 用户："检测所有 person 和 dog"
      → categories="person,dog"

    - 用户："有没有穿 red clothes 的 person"
      → categories="person wearing red clothes" 

    - 用户："找一下坐在 chair 上看书的人"
      → categories="person sitting on chair reading book"

    复杂场景示例：
    - 用户："提示标语玻璃范围里最黄的树"
      → categories="提示标语玻璃范围里最黄的树"

    - 用户："我在和企鹅合照，你知道我在哪里吗"
      → categories="与企鹅合照的人"

    - 用户："帮我找一下坐在椅子上看书的人"
      → categories="坐在椅子上看书的人"

    最佳实践：
    - 保持与用户输入语言的一致性
    - 使用简洁但完整的描述
    - 保留能区分目标的关键属性（颜色、状态、位置等）
    - 多个目标用逗号分隔
    - 每个类别应该是一个可独立检测的视觉概念
    - 不要包含非视觉信息（如数量"几个"、疑问词"是否"等）
    - 不要过度简化导致丢失关键信息

    返回:
    包含检测结果的字典，包括坐标、数量、图片尺寸等信息
    """
    image_bytes = _image_context.get("bytes")
    image_filename = _image_context.get("filename")

    if not image_bytes:
        return {"error": "没有找到图片，请先上传图片"}

    print(f"🔍 检测图片: {image_filename}")
    print(f"   检测类别: {categories}")

    files = {'image': (image_filename, image_bytes, 'image/jpeg')}
    data = {
        'categories': categories,
        'return_visualization': True,  # 强制返回可视化图片
        'show_labels': True
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                "http://localhost:8000/api/detect_for_chat",
                files=files,
                data=data
            )
            resp.raise_for_status()
            result = resp.json()

        print(f"📦 API 返回: success={result.get('success')}")

        # 提取数据
        visualization_b64 = result.get("visualization")
        detection_results = result.get("detection_results", {})

        # 统计检测数量
        total_count = sum(len(objs) for objs in detection_results.values() if isinstance(objs, list))
        print(f"   检测到 {total_count} 个对象，类别: {list(detection_results.keys())}")

        # 存储到上下文（供后处理节点使用）
        _image_context["visualization_b64"] = visualization_b64
        _image_context["detection_results"] = detection_results

        # 工具返回：只返回坐标，不返回图片
        return {
            "detection_results": detection_results,
            "count": total_count,
            "image_size": result.get("image_size"),
            "inference_time": result.get("inference_time")
        }

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ 检测错误详情:\n{error_detail}")
        return {"error": f"检测服务错误: {str(e)}"}

# ====================== 模型配置 ======================
text_llm = ChatQwen(
    model="qwen3-vl-plus",
    max_tokens=8192,
    temperature=0.7,
    timeout=20,
    api_key=settings.DASHSCOPE_API_KEY,
    base_url=settings.DASHSCOPE_BASE_URL,
).bind_tools([detect_objects]) 

realtime_llm = ChatQwen(
    model="qwen3-omni-turbo-realtime",
    max_tokens=16384,
    temperature=0.7,
    timeout=5,
    streaming=True,
    api_key=settings.DASHSCOPE_API_KEY,
    base_url=settings.DASHSCOPE_BASE_URL,
)

# ====================== 预处理节点（修复版）======================
def preprocess_node(state: AgentState):
    """在首次调用前，将图片添加到用户消息中"""
    messages = state["messages"]
    image_bytes = state.get("image_bytes")
    image_filename = state.get("image_filename")
    
    # 更新工具上下文
    _image_context["bytes"] = image_bytes
    _image_context["filename"] = image_filename
    
    print(f"🔧 预处理: 消息数={len(messages)}, 有图片={bool(image_bytes)}")
    
    # 如果有图片且有用户消息
    if image_bytes and messages:
        # 找到第一条用户消息
        for msg in messages:
            if isinstance(msg, HumanMessage):
                # 检查是否已包含图片
                if isinstance(msg.content, list):
                    has_image = any(
                        isinstance(item, dict) and item.get("type") == "image_url" 
                        for item in msg.content
                    )
                    if has_image:
                        print("   图片已存在，跳过")
                        return {}
                
                # 编码图片
                image_b64 = base64.b64encode(image_bytes).decode()
                
                # 构造新内容
                if isinstance(msg.content, str):
                    new_content = [
                        {"type": "text", "text": msg.content},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                        }
                    ]
                else:
                    new_content = list(msg.content) + [{
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    }]
                
                # ✅ 使用相同的 ID 替换消息，而不是追加
                print(f"✅ 替换消息（ID: {msg.id}）添加图片")
                return {
                    "messages": [HumanMessage(content=new_content, id=msg.id)]
                }
    
    return {}

# ====================== Agent 节点 ======================
def agent_node(state: AgentState):
    """调用 LLM"""
    messages = state["messages"]
    model_type = state.get("model_type", "text")

    print(f"\n🤖 Agent 节点: 消息数={len(messages)}")
    for idx, msg in enumerate(messages):
        role = msg.__class__.__name__
        if isinstance(msg.content, list):
            types = [item.get("type", "?") for item in msg.content if isinstance(item, dict)]
            print(f"   [{idx}] {role} (ID: {msg.id[:8]}...): {types}")
        else:
            content_preview = str(msg.content)[:50].replace('\n', ' ')
            print(f"   [{idx}] {role} (ID: {msg.id[:8]}...): {content_preview}...")

    # 选择模型
    llm = text_llm if model_type == "text" else realtime_llm

    # 调用 LLM
    try:
        response = llm.invoke(messages)
        print(f"✅ LLM 响应: {response.__class__.__name__}\n")
        return {"messages": [response]}
    except Exception as e:
        print(f"❌ LLM 失败: {e}\n")
        raise

# ====================== 后处理节点（将带框图片注入消息）======================
def post_tools_node(state: AgentState):
    """在工具调用后，将带框图片添加到消息中"""
    _ = state  # 保留参数以符合节点签名
    visualization_b64 = _image_context.get("visualization_b64")

    if not visualization_b64:
        print("⚠️ 没有检测图片，跳过后处理")
        return {}

    print(f"📸 添加带框图片到消息流")

    # 处理base64格式（去除可能的前缀）
    if visualization_b64.startswith("data:"):
        image_url = visualization_b64  # 已经是完整URL
    else:
        # 纯base64，添加前缀
        image_url = f"data:image/jpeg;base64,{visualization_b64}"

    # 构造包含图片的消息
    annotated_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "这是检测后带标注框的图片，请基于这张图片和上面的坐标信息回答问题："
            },
            {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
        ]
    )

    # 注意：不清除 visualization_b64，供 web_server 返回给前端使用
    # 下次检测时会自动覆盖

    return {"messages": [annotated_message]}

# ====================== 创建图 ======================
def create_graph(model_type: Literal["text", "realtime"]):
    builder = StateGraph(AgentState)

    tools = [detect_objects] if model_type == "text" else []

    # 添加节点
    builder.add_node("preprocess", preprocess_node)
    builder.add_node("agent", agent_node)

    if tools:
        builder.add_node("tools", ToolNode(tools))
        builder.add_node("post_tools", post_tools_node)  # 后处理节点

    # 连接节点
    builder.add_edge(START, "preprocess")
    builder.add_edge("preprocess", "agent")

    if tools:
        # LLM 自己判断是否需要调用工具
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "post_tools")      # tools -> post_tools
        builder.add_edge("post_tools", "agent")       # post_tools -> agent
    else:
        builder.add_edge("agent", END)

    memory = MemorySaver()
    # return builder.compile()
    return builder.compile(checkpointer=memory)

text_graph = create_graph("text")
realtime_graph = create_graph("realtime")

# ====================== 统一入口 ======================
class DynamicQwenAgent:
    def __init__(self):
        self.graphs = {
            "text": text_graph,
            "realtime": realtime_graph
        }
    
    def invoke(self, input: dict, config: RunnableConfig | None = None):
        config = config or {}
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        model_type = config.get("model_type", "text")
        
        if model_type not in self.graphs:
            raise ValueError(f"不支持的 model_type: {model_type}")
        
        graph = self.graphs[model_type]

        # 构造输入
        messages = list(input.get("messages", []))
        if "input" in input:
            messages.append(HumanMessage(content=input["input"]))
        
        print(f"\n{'='*60}")
        print(f"🚀 开始执行: thread_id={thread_id}, model={model_type}")
        print(f"📝 输入消息: {len(messages)} 条")
        print(f"{'='*60}\n")
        
        full_input = {
            "messages": messages,
            "image_bytes": input.get("image_bytes"),        
            "image_filename": input.get("image_filename"),  
            "model_type": model_type
        }

        result = graph.invoke(full_input, {"configurable": {"thread_id": thread_id}})
        
        # 提取最后的 AI 消息
        last_ai_message = None
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                last_ai_message = msg
                break

        output = last_ai_message.content if last_ai_message else "（无回复）"
        
        print(f"\n{'='*60}")
        print(f"✅ 执行完成")
        print(f"📊 最终消息数: {len(result.get('messages', []))}")
        print(f"{'='*60}\n")
        
        return {"output": output}
    
    async def ainvoke(self, input: dict, config: RunnableConfig | None = None):
        return self.invoke(input, config)
    
chat_agent = DynamicQwenAgent()