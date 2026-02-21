"""
定时任务工具 (Cron tool for scheduling reminders and tasks)

提供定时任务调度功能，支持三种调度方式:
- every_seconds: 固定间隔执行（如每 60 秒）
- cron_expr: Cron 表达式（如 "0 9 * * *" 每天 9 点）
- at: 一次性定时执行（如 "2026-02-12T10:30:00"）

LLM 调用示例:
- 添加任务: {"action": "add", "message": "喝水提醒", "every_seconds": 3600}
- 列出任务: {"action": "list"}
- 删除任务: {"action": "remove", "job_id": "abc123"}
"""

from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule


class CronTool(Tool):
    """
    定时任务调度工具 (Tool to schedule reminders and recurring tasks)
    
    支持的操作:
    - add: 添加定时任务
    - list: 列出所有任务
    - remove: 删除指定任务
    """
    
    def __init__(self, cron_service: CronService):
        # 定时任务服务（由 CronService 管理任务的存储和执行）
        self._cron = cron_service
        # 当前会话的渠道和聊天 ID（用于任务触发时发送消息）
        self._channel = ""
        self._chat_id = ""
    
    def set_context(self, channel: str, chat_id: str) -> None:
        """
        设置当前会话上下文 (Set the current session context for delivery)
        
        任务触发时会向这个渠道/聊天发送消息。
        由 AgentLoop._set_tool_context() 在处理每条消息时调用。
        """
        self._channel = channel
        self._chat_id = chat_id
    
    @property
    def name(self) -> str:
        return "cron"
    
    @property
    def description(self) -> str:
        return "Schedule reminders and recurring tasks. Actions: add, list, remove."
    
    @property
    def parameters(self) -> dict[str, Any]:
        # OpenAI 函数调用格式的参数定义
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "remove"],
                    "description": "Action to perform"
                },
                "message": {
                    "type": "string",
                    "description": "Reminder message (for add)"
                },
                "every_seconds": {
                    "type": "integer",
                    "description": "Interval in seconds (for recurring tasks)"
                },
                "cron_expr": {
                    "type": "string",
                    "description": "Cron expression like '0 9 * * *' (for scheduled tasks)"
                },
                "tz": {
                    "type": "string",
                    "description": "IANA timezone for cron expressions (e.g. 'America/Vancouver')"
                },
                "at": {
                    "type": "string",
                    "description": "ISO datetime for one-time execution (e.g. '2026-02-12T10:30:00')"
                },
                "job_id": {
                    "type": "string",
                    "description": "Job ID (for remove)"
                }
            },
            "required": ["action"]
        }
    
    async def execute(
        self,
        action: str,
        message: str = "",
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        tz: str | None = None,
        at: str | None = None,
        job_id: str | None = None,
        **kwargs: Any
    ) -> str:
        """
        执行定时任务操作 (Execute cron action)
        
        根据 action 参数分发到对应的处理方法。
        """
        if action == "add":
            return self._add_job(message, every_seconds, cron_expr, tz, at)
        elif action == "list":
            return self._list_jobs()
        elif action == "remove":
            return self._remove_job(job_id)
        return f"Unknown action: {action}"
    
    def _add_job(
        self,
        message: str,
        every_seconds: int | None,
        cron_expr: str | None,
        tz: str | None,
        at: str | None,
    ) -> str:
        """
        添加定时任务 (Add a scheduled job)
        
        支持三种调度方式（互斥）:
        1. every_seconds: 固定间隔执行
        2. cron_expr: Cron 表达式（可配合 tz 时区）
        3. at: 一次性定时执行（执行后自动删除）
        """
        # ========== 参数验证 ==========
        if not message:
            return "Error: message is required for add"
        if not self._channel or not self._chat_id:
            return "Error: no session context (channel/chat_id)"
        # tz 只能与 cron_expr 一起使用
        if tz and not cron_expr:
            return "Error: tz can only be used with cron_expr"
        # 验证时区是否有效
        if tz:
            from zoneinfo import ZoneInfo
            try:
                ZoneInfo(tz)
            except (KeyError, Exception):
                return f"Error: unknown timezone '{tz}'"
        
        # ========== 构建调度配置 ==========
        delete_after = False  # 是否执行后删除
        if every_seconds:
            # 固定间隔执行（转换为毫秒）
            schedule = CronSchedule(kind="every", every_ms=every_seconds * 1000)
        elif cron_expr:
            # Cron 表达式（如 "0 9 * * *" 表示每天 9 点）
            schedule = CronSchedule(kind="cron", expr=cron_expr, tz=tz)
        elif at:
            # 一次性定时执行
            from datetime import datetime
            dt = datetime.fromisoformat(at)
            at_ms = int(dt.timestamp() * 1000)
            schedule = CronSchedule(kind="at", at_ms=at_ms)
            delete_after = True  # 一次性任务执行后自动删除
        else:
            return "Error: either every_seconds, cron_expr, or at is required"
        
        # ========== 创建任务 ==========
        job = self._cron.add_job(
            name=message[:30],           # 任务名称（截取前 30 字符）
            schedule=schedule,           # 调度配置
            message=message,             # 完整消息内容
            deliver=True,                # 触发时发送消息
            channel=self._channel,       # 目标渠道
            to=self._chat_id,            # 目标聊天 ID
            delete_after_run=delete_after,  # 是否执行后删除
        )
        return f"Created job '{job.name}' (id: {job.id})"
    
    def _list_jobs(self) -> str:
        """列出所有定时任务 (List all scheduled jobs)"""
        jobs = self._cron.list_jobs()
        if not jobs:
            return "No scheduled jobs."
        # 格式化输出：名称、ID、调度类型
        lines = [f"- {j.name} (id: {j.id}, {j.schedule.kind})" for j in jobs]
        return "Scheduled jobs:\n" + "\n".join(lines)
    
    def _remove_job(self, job_id: str | None) -> str:
        """删除指定的定时任务 (Remove a scheduled job by ID)"""
        if not job_id:
            return "Error: job_id is required for remove"
        if self._cron.remove_job(job_id):
            return f"Removed job {job_id}"
        return f"Job {job_id} not found"
