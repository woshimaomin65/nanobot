"""
文件系统工具 (File system tools: read, write, edit)

提供文件读写和目录操作的工具集，供 LLM 调用。

包含的工具:
- ReadFileTool: 读取文件内容
- WriteFileTool: 写入文件内容
- EditFileTool: 编辑文件（查找替换）
- ListDirTool: 列出目录内容

安全特性:
- 支持 allowed_dir 参数限制文件操作范围
- 路径解析时自动展开 ~ 和解析符号链接
"""

from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


def _resolve_path(path: str, allowed_dir: Path | None = None) -> Path:
    """
    解析路径并可选地强制目录限制 (Resolve path and optionally enforce directory restriction)
    
    Args:
        path: 用户提供的路径字符串
        allowed_dir: 可选的允许目录，如果设置则路径必须在此目录内
    
    Returns:
        解析后的绝对路径
    
    Raises:
        PermissionError: 如果路径在允许目录之外
    """
    # expanduser(): 展开 ~ 为用户主目录
    # resolve(): 解析符号链接并返回绝对路径
    resolved = Path(path).expanduser().resolve()
    # 安全检查：确保路径在允许的目录内
    if allowed_dir and not str(resolved).startswith(str(allowed_dir.resolve())):
        raise PermissionError(f"Path {path} is outside allowed directory {allowed_dir}")
    return resolved


# ==================== ReadFileTool ====================
class ReadFileTool(Tool):
    """
    读取文件内容工具 (Tool to read file contents)
    
    LLM 调用示例:
    {"name": "read_file", "arguments": {"path": "src/main.py"}}
    """
    
    def __init__(self, allowed_dir: Path | None = None):
        # 允许的目录限制（用于沙箱模式）
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return "Read the contents of a file at the given path."
    
    @property
    def parameters(self) -> dict[str, Any]:
        # OpenAI 函数调用格式的参数定义
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to read"
                }
            },
            "required": ["path"]
        }
    
    async def execute(self, path: str, **kwargs: Any) -> str:
        """执行文件读取操作"""
        try:
            # 解析并验证路径
            file_path = _resolve_path(path, self._allowed_dir)
            if not file_path.exists():
                return f"Error: File not found: {path}"
            if not file_path.is_file():
                return f"Error: Not a file: {path}"
            
            # 读取文件内容（UTF-8 编码）
            content = file_path.read_text(encoding="utf-8")
            return content
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading file: {str(e)}"


# ==================== WriteFileTool ====================
class WriteFileTool(Tool):
    """
    写入文件内容工具 (Tool to write content to a file)
    
    特性:
    - 自动创建父目录
    - 覆盖已存在的文件
    
    LLM 调用示例:
    {"name": "write_file", "arguments": {"path": "output.txt", "content": "Hello World"}}
    """
    
    def __init__(self, allowed_dir: Path | None = None):
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "write_file"
    
    @property
    def description(self) -> str:
        return "Write content to a file at the given path. Creates parent directories if needed."
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to write to"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write"
                }
            },
            "required": ["path", "content"]
        }
    
    async def execute(self, path: str, content: str, **kwargs: Any) -> str:
        """执行文件写入操作"""
        try:
            file_path = _resolve_path(path, self._allowed_dir)
            # 自动创建父目录（如果不存在）
            file_path.parent.mkdir(parents=True, exist_ok=True)
            # 写入文件内容
            file_path.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to {path}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error writing file: {str(e)}"


# ==================== EditFileTool ====================
class EditFileTool(Tool):
    """
    编辑文件工具 - 通过查找替换修改文件 (Tool to edit a file by replacing text)
    
    特性:
    - 精确匹配 old_text
    - 如果 old_text 出现多次，要求用户提供更多上下文
    - 只替换第一次出现
    
    LLM 调用示例:
    {"name": "edit_file", "arguments": {"path": "config.py", "old_text": "DEBUG = False", "new_text": "DEBUG = True"}}
    """
    
    def __init__(self, allowed_dir: Path | None = None):
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "edit_file"
    
    @property
    def description(self) -> str:
        return "Edit a file by replacing old_text with new_text. The old_text must exist exactly in the file."
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to edit"
                },
                "old_text": {
                    "type": "string",
                    "description": "The exact text to find and replace"
                },
                "new_text": {
                    "type": "string",
                    "description": "The text to replace with"
                }
            },
            "required": ["path", "old_text", "new_text"]
        }
    
    async def execute(self, path: str, old_text: str, new_text: str, **kwargs: Any) -> str:
        """执行文件编辑操作"""
        try:
            file_path = _resolve_path(path, self._allowed_dir)
            if not file_path.exists():
                return f"Error: File not found: {path}"
            
            # 读取当前文件内容
            content = file_path.read_text(encoding="utf-8")
            
            # 检查 old_text 是否存在
            if old_text not in content:
                return f"Error: old_text not found in file. Make sure it matches exactly."
            
            # 检查 old_text 出现次数（防止歧义替换）
            count = content.count(old_text)
            if count > 1:
                return f"Warning: old_text appears {count} times. Please provide more context to make it unique."
            
            # 执行替换（只替换第一次出现）
            new_content = content.replace(old_text, new_text, 1)
            file_path.write_text(new_content, encoding="utf-8")
            
            return f"Successfully edited {path}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error editing file: {str(e)}"


# ==================== ListDirTool ====================
class ListDirTool(Tool):
    """
    列出目录内容工具 (Tool to list directory contents)
    
    输出格式:
    - 📁 表示目录
    - 📄 表示文件
    - 按名称排序
    
    LLM 调用示例:
    {"name": "list_dir", "arguments": {"path": "src"}}
    """
    
    def __init__(self, allowed_dir: Path | None = None):
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "list_dir"
    
    @property
    def description(self) -> str:
        return "List the contents of a directory."
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory path to list"
                }
            },
            "required": ["path"]
        }
    
    async def execute(self, path: str, **kwargs: Any) -> str:
        """执行目录列表操作"""
        try:
            dir_path = _resolve_path(path, self._allowed_dir)
            if not dir_path.exists():
                return f"Error: Directory not found: {path}"
            if not dir_path.is_dir():
                return f"Error: Not a directory: {path}"
            
            # 遍历目录内容并格式化输出
            items = []
            for item in sorted(dir_path.iterdir()):
                # 使用 emoji 区分目录和文件
                prefix = "📁 " if item.is_dir() else "📄 "
                items.append(f"{prefix}{item.name}")
            
            if not items:
                return f"Directory {path} is empty"
            
            return "\n".join(items)
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error listing directory: {str(e)}"
