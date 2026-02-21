"""
子代理管理器 - 用于后台任务执行 (Subagent manager for background task execution)

子代理是一种轻量级的代理实例，在后台运行以处理特定任务。
它们与主代理共享同一个 LLM provider，但拥有独立的上下文和专注的系统提示。

主要功能:
- spawn(): 创建并启动一个后台子代理
- _run_subagent(): 执行子代理的核心循环
- _announce_result(): 将结果通过消息总线发送回主代理

调用链:
    主代理 (AgentLoop)
        └── SpawnTool.execute()
                └── SubagentManager.spawn()
                        └── _run_subagent() [asyncio.Task]
                                └── _announce_result() → MessageBus → 主代理
"""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool


class SubagentManager:
    """
    子代理管理器 (Manages background subagent execution)
    
    子代理是轻量级的代理实例，在后台运行以处理特定任务。
    它们与主代理共享同一个 LLM provider，但拥有独立的上下文和专注的系统提示。
    
    与主代理的区别:
    - 没有 message 工具（不能直接发消息给用户）
    - 没有 spawn 工具（不能再创建子代理）
    - 迭代次数限制更严格（15次 vs 主代理的更多）
    - 完成后通过消息总线通知主代理
    """
    
    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
    ):
        """
        初始化子代理管理器。
        
        Args:
            provider: LLM 提供者，与主代理共享
            workspace: 工作区路径
            bus: 消息总线，用于将结果发送回主代理
            model: 使用的模型名称
            temperature: 生成温度
            max_tokens: 最大 token 数
            brave_api_key: Brave 搜索 API 密钥
            exec_config: 命令执行配置
            restrict_to_workspace: 是否限制文件操作在工作区内
        """
        from nanobot.config.schema import ExecToolConfig
        # LLM 提供者（与主代理共享）
        self.provider = provider
        # 工作区路径
        self.workspace = workspace
        # 消息总线（用于将结果发送回主代理）
        self.bus = bus
        # 模型配置
        self.model = model or provider.get_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        # 正在运行的子代理任务字典 {task_id: asyncio.Task}
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
    
    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
    ) -> str:
        """
        创建并启动一个后台子代理 (Spawn a subagent to execute a task in the background)
        
        这是子代理的入口点，由 SpawnTool 调用。
        
        Args:
            task: 子代理要执行的任务描述
            label: 可选的人类可读标签
            origin_channel: 结果发送的目标渠道
            origin_chat_id: 结果发送的目标聊天 ID
        
        Returns:
            状态消息，表示子代理已启动
        """
        # 生成唯一的任务 ID（取 UUID 前 8 位）
        task_id = str(uuid.uuid4())[:8]
        # 显示标签：使用提供的标签或截取任务描述前 30 字符
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        
        # 记录结果发送的目标位置
        origin = {
            "channel": origin_channel,
            "chat_id": origin_chat_id,
        }
        
        # ========== 创建后台异步任务 ==========
        # 使用 asyncio.create_task 在后台运行子代理
        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin)
        )
        # 将任务添加到运行中的任务字典
        self._running_tasks[task_id] = bg_task
        
        # 任务完成时自动从字典中移除
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(task_id, None))
        
        logger.info(f"Spawned subagent [{task_id}]: {display_label}")
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."
    
    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
    ) -> None:
        """
        执行子代理任务并发布结果 (Execute the subagent task and announce the result)
        
        这是子代理的核心执行循环，类似于主代理的 _run_agent_loop，但更简化。
        
        流程:
        1. 注册工具（不包含 message 和 spawn）
        2. 构建系统提示和初始消息
        3. 运行 ReAct 循环（最多 15 次迭代）
        4. 将结果通过消息总线发送回主代理
        """
        logger.info(f"Subagent [{task_id}] starting task: {label}")
        
        try:
            # ========== 第1步：注册子代理可用的工具 ==========
            # 注意：子代理没有 message 工具和 spawn 工具
            tools = ToolRegistry()
            allowed_dir = self.workspace if self.restrict_to_workspace else None
            # 文件操作工具
            tools.register(ReadFileTool(allowed_dir=allowed_dir))
            tools.register(WriteFileTool(allowed_dir=allowed_dir))
            tools.register(EditFileTool(allowed_dir=allowed_dir))
            tools.register(ListDirTool(allowed_dir=allowed_dir))
            # Shell 执行工具
            tools.register(ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
            ))
            # Web 工具
            tools.register(WebSearchTool(api_key=self.brave_api_key))
            tools.register(WebFetchTool())
            
            # ========== 第2步：构建初始消息 ==========
            # 使用专门的子代理系统提示
            system_prompt = self._build_subagent_prompt(task)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]
            
            # ========== 第3步：运行 ReAct 循环 ==========
            # 子代理的迭代次数限制为 15 次
            max_iterations = 15
            iteration = 0
            final_result: str | None = None
            
            while iteration < max_iterations:
                iteration += 1
                
                # 调用 LLM
                response = await self.provider.chat(
                    messages=messages,
                    tools=tools.get_definitions(),
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                if response.has_tool_calls:
                    # ---------- 处理工具调用 ----------
                    # 构建工具调用字典
                    tool_call_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ]
                    # 添加助手消息
                    messages.append({
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": tool_call_dicts,
                    })
                    
                    # 执行每个工具调用
                    for tool_call in response.tool_calls:
                        args_str = json.dumps(tool_call.arguments)
                        logger.debug(f"Subagent [{task_id}] executing: {tool_call.name} with arguments: {args_str}")
                        result = await tools.execute(tool_call.name, tool_call.arguments)
                        # 添加工具结果
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result,
                        })
                else:
                    # ---------- 无工具调用，任务完成 ----------
                    final_result = response.content
                    break
            
            # 如果达到最大迭代次数但没有最终结果
            if final_result is None:
                final_result = "Task completed but no final response was generated."
            
            # ========== 第4步：发布结果 ==========
            logger.info(f"Subagent [{task_id}] completed successfully")
            await self._announce_result(task_id, label, task, final_result, origin, "ok")
            
        except Exception as e:
            # 错误处理：将错误信息发送回主代理
            error_msg = f"Error: {str(e)}"
            logger.error(f"Subagent [{task_id}] failed: {e}")
            await self._announce_result(task_id, label, task, error_msg, origin, "error")
    
    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
    ) -> None:
        """
        通过消息总线将子代理结果发送给主代理 (Announce the subagent result to the main agent via the message bus)
        
        这是子代理与主代理通信的唯一方式。
        结果会被包装成一个 InboundMessage，注入到消息总线中，
        主代理会像处理普通用户消息一样处理它。
        """
        status_text = "completed successfully" if status == "ok" else "failed"
        
        # 构建发送给主代理的消息内容
        # 包含任务描述、结果，以及让主代理总结的指令
        announce_content = f"""[Subagent '{label}' {status_text}]

Task: {task}

Result:
{result}

Summarize this naturally for the user. Keep it brief (1-2 sentences). Do not mention technical details like "subagent" or task IDs."""
        
        # 创建入站消息，注入到消息总线
        # channel="system" 表示这是系统内部消息
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )
        
        # 发布到消息总线，主代理会在下一次循环中处理
        await self.bus.publish_inbound(msg)
        logger.debug(f"Subagent [{task_id}] announced result to {origin['channel']}:{origin['chat_id']}")
    
    def _build_subagent_prompt(self, task: str) -> str:
        """
        构建子代理的系统提示 (Build a focused system prompt for the subagent)
        
        子代理的系统提示比主代理更简洁，专注于单一任务。
        """
        from datetime import datetime
        import time as _time
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"

        return f"""# Subagent

## Current Time
{now} ({tz})

You are a subagent spawned by the main agent to complete a specific task.

## Rules
1. Stay focused - complete only the assigned task, nothing else
2. Your final response will be reported back to the main agent
3. Do not initiate conversations or take on side tasks
4. Be concise but informative in your findings

## What You Can Do
- Read and write files in the workspace
- Execute shell commands
- Search the web and fetch web pages
- Complete the task thoroughly

## What You Cannot Do
- Send messages directly to users (no message tool available)
- Spawn other subagents
- Access the main agent's conversation history

## Workspace
Your workspace is at: {self.workspace}
Skills are available at: {self.workspace}/skills/ (read SKILL.md files as needed)

When you have completed the task, provide a clear summary of your findings or actions."""
    
    def get_running_count(self) -> int:
        """返回当前正在运行的子代理数量 (Return the number of currently running subagents)"""
        return len(self._running_tasks)
