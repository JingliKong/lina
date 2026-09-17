"""Base tool class and built-in tools."""

import subprocess
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

from bs4 import BeautifulSoup


class Tool(ABC):
    """Base class for all tools."""

    name: str
    description: str
    parameters: dict  # JSON Schema
    requires_permission: bool = False

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool and return the result."""
        pass

# Where full (untruncated) outputs get stashed. Keep this outside the
# repo you're operating on so it never shows up in grep/glob results.
_BLOB_DIR = Path.home() / ".henri" / "tool_output_blobs"

MAX_LINES = 200          # per stream (stdout / stderr), after shaping
MAX_LINE_LEN = 2000      # guard against single absurdly long lines
                          # (e.g. a minified JS bundle or base64 blob)

def _shape_stream(text: str, label: str, blob_id: str) -> str:
    """Cap a single stream by line count, saving the full text to a
    blob file and leaving a ref the model can ask for by id."""
    if not text:
        return ""

    lines = text.splitlines()

    # Clip any individual absurd line before counting/joining, so one
    # 500KB line doesn't defeat the line-count cap.
    lines = [
        (ln if len(ln) <= MAX_LINE_LEN else ln[:MAX_LINE_LEN] + " …[line truncated]")
        for ln in lines
    ]

    if len(lines) <= MAX_LINES:
        return "\n".join(lines)

    _BLOB_DIR.mkdir(parents=True, exist_ok=True)
    blob_path = _BLOB_DIR / f"{blob_id}_{label}.txt"
    blob_path.write_text(text)

    shown = lines[:MAX_LINES]
    omitted = len(lines) - MAX_LINES
    return (
        "\n".join(shown)
        + f"\n[{label}: {omitted} more line(s) omitted — "
        + f"full output saved to {blob_path}. "
        + f"Use read_file(path=\"{blob_path}\", offset=...) to page through it "
        + "if these lines aren't enough — don't re-run the command.]"
    )


class BashTool(Tool):
    """Execute shell commands."""

    name = "bash"
    description = (
        "Execute a shell command and return its output. Output is capped "
        f"at {MAX_LINES} lines per stream (stdout/stderr); if the command "
        "produces more, the full output is saved to disk and a path is "
        "given so you can read specific parts instead of re-running it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
        },
        "required": ["command"],
    }
    requires_permission = True

    def execute(self, command: str) -> str:
        blob_id = f"{int(time.time() * 1000)}"
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return "[error: command timed out after 120 seconds]"
        except Exception as e:
            return f"[error: {e}]"

        stdout = _shape_stream(result.stdout, "stdout", blob_id)
        stderr = _shape_stream(result.stderr, "stderr", blob_id)

        parts = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"[stderr]\n{stderr}")
        if result.returncode != 0:
            parts.append(f"[exit code: {result.returncode}]")

        return "\n".join(parts) if parts else "(no output)"

class ReadFileTool(Tool):
    """Read file contents."""

    name = "read_file"
    description = "Read the contents of a file."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read",
            },
        },
        "required": ["path"],
    }
    requires_permission = True  # Permission managed by path (auto-allow within cwd)

    def execute(self, path: str) -> str:
        try:
            p = Path(path).expanduser()
            if not p.exists():
                return f"[error: file not found: {path}]"
            if not p.is_file():
                return f"[error: not a file: {path}]"
            content = p.read_text()
            if len(content) > 100_000:
                return content[:100_000] + "\n[truncated...]"
            return content
        except Exception as e:
            return f"[error: {e}]"


class WriteFileTool(Tool):
    """Write content to a file."""

    name = "write_file"
    description = "Write content to a file. Creates the file if it doesn't exist."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
        },
        "required": ["path", "content"],
    }
    requires_permission = True

    def execute(self, path: str, content: str) -> str:
        try:
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return f"[wrote {len(content)} bytes to {path}]"
        except Exception as e:
            return f"[error: {e}]"


class EditFileTool(Tool):
    """Edit a file by replacing exact text."""

    name = "edit_file"
    description = (
        "Replace exact text in a file. The old_string must be unique in the file "
        "(or use replace_all=true to replace all occurrences). "
        "Include enough context in old_string to make it unique."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit",
            },
            "old_string": {
                "type": "string",
                "description": "The exact text to find and replace",
            },
            "new_string": {
                "type": "string",
                "description": "The text to replace it with",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences instead of just the first",
                "default": False,
            },
        },
        "required": ["path", "old_string", "new_string"],
    }
    requires_permission = True

    def execute(
        self, path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        try:
            p = Path(path).expanduser()
            if not p.exists():
                return f"[error: file not found: {path}]"
            if not p.is_file():
                return f"[error: not a file: {path}]"

            content = p.read_text()
            count = content.count(old_string)

            if count == 0:
                return f"[error: old_string not found in {path}]"
            if count > 1 and not replace_all:
                return (
                    f"[error: old_string appears {count} times in {path}. "
                    f"Use replace_all=true or provide more context to make it unique.]"
                )

            if replace_all:
                new_content = content.replace(old_string, new_string)
                replacements = count
            else:
                new_content = content.replace(old_string, new_string, 1)
                replacements = 1

            p.write_text(new_content)
            return f"[replaced {replacements} occurrence(s) in {path}]"
        except Exception as e:
            return f"[error: {e}]"


class GrepTool(Tool):
    """Search for patterns in files using ripgrep."""

    name = "grep"
    description = (
        "Search for a regex pattern in files using grep"
        "Returns matching lines with file paths and line numbers."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search in (default: current directory)",
                "default": ".",
            },
            "glob": {
                "type": "string",
                "description": "Only search files matching this glob pattern (e.g., '*.py')",
            },
            "ignore_case": {
                "type": "boolean",
                "description": "Case-insensitive search",
                "default": False,
            },
        },
        "required": ["pattern"],
    }
    requires_permission = False  # Permission managed by path (auto-allow within cwd)

    def execute(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        ignore_case: bool = False,
    ) -> str:
        try:
            cmd = ["rg", "--line-number", "--max-count", "100"]
            if ignore_case:
                cmd.append("--ignore-case")
            if glob:
                cmd.extend(["--glob", glob])
            cmd.extend([pattern, path])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout
            if result.returncode == 1:  # No matches
                return "(no matches)"
            if result.returncode != 0:
                return f"[error: {result.stderr}]"
            if len(output) > 50_000:
                output = output[:50_000] + "\n[truncated...]"
            return output or "(no matches)"
        except FileNotFoundError:
            return "[error: ripgrep (rg) not found. Install it: brew install ripgrep]"
        except subprocess.TimeoutExpired:
            return "[error: search timed out after 30 seconds]"
        except Exception as e:
            return f"[error: {e}]"


class GlobTool(Tool):
    """Find files matching a glob pattern."""

    name = "glob"
    description = "Find files matching a glob pattern (e.g., '**/*.py', 'src/**/*.ts')."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The glob pattern to match (e.g., '**/*.py')",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: current directory)",
                "default": ".",
            },
        },
        "required": ["pattern"],
    }
    requires_permission = True  # Permission managed by path (auto-allow within cwd)

    def execute(self, pattern: str, path: str = ".") -> str:
        try:
            p = Path(path).expanduser()
            if not p.exists():
                return f"[error: directory not found: {path}]"
            if not p.is_dir():
                return f"[error: not a directory: {path}]"

            matches = sorted(p.glob(pattern))[:100]
            if not matches:
                return "(no matches)"
            return "\n".join(str(m) for m in matches)
        except Exception as e:
            return f"[error: {e}]"


class WebFetchTool(Tool):
    """Fetch content from a URL."""

    name = "web_fetch"
    description = "Fetch content from a URL and return the text. HTML is converted to plain text."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch",
            },
        },
        "required": ["url"],
    }
    # requires_permission = True  # Network access requires permission
    requires_permission = False   # IDK I like searching.

    def execute(self, url: str) -> str:
        try:
            # Ensure URL has a scheme
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Henri/0.1 (AI coding assistant)"},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                content_type = response.headers.get("Content-Type", "")
                content = response.read().decode("utf-8", errors="replace")

                # Convert HTML to text
                if "html" in content_type.lower():
                    soup = BeautifulSoup(content, "html.parser")
                    for tag in soup(["script", "style", "head"]):
                        tag.decompose()
                    content = soup.get_text(separator="\n", strip=True)

                if len(content) > 50_000:
                    content = content[:50_000] + "\n[truncated...]"

                return content or "(empty response)"
        except urllib.error.HTTPError as e:
            return f"[error: HTTP {e.code} {e.reason}]"
        except urllib.error.URLError as e:
            return f"[error: {e.reason}]"
        except Exception as e:
            return f"[error: {e}]"


def get_default_tools() -> list[Tool]:
    """Return the default set of tools."""
    return [
        BashTool(),
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        GrepTool(),
        GlobTool(),
        WebFetchTool(),
    ]
