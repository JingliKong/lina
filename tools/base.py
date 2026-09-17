"""Base tool class and built-in tools."""

import subprocess
import time
import urllib.request
import urllib.error
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


# ---------------------------------------------------------------------------
# Shared output-shaping helpers
#
# The guiding rule: a tool result should never enter the message list at
# full size. Cap by *lines* (not characters) so truncation never slices a
# line in half, and always tell the model what it isn't seeing plus how to
# get more — a silent cutoff makes the model reason confidently over a
# partial view, which is worse than a big transcript.
# ---------------------------------------------------------------------------

# Where full (untruncated) bash outputs get stashed. Keep this outside the
# repo you're operating on so it never shows up in grep/glob results.
_BLOB_DIR = Path.home() / ".henri" / "tool_output_blobs"

MAX_LINES = 200          # per stream (stdout / stderr), after shaping
MAX_LINE_LEN = 2000      # guard against single absurdly long lines
                         # (e.g. a minified JS bundle or base64 blob)


def _clip_line(line: str) -> str:
    """Clip one absurdly long line so it can't defeat a line-count cap."""
    if len(line) <= MAX_LINE_LEN:
        return line
    return line[:MAX_LINE_LEN] + " …[line truncated]"


def _shape_stream(text: str, label: str, blob_id: str) -> str:
    """Cap a single stream by line count, saving the full text to a blob
    file and leaving a path the model can page through instead of
    re-running the command (which may have side effects)."""
    if not text:
        return ""

    lines = [_clip_line(ln) for ln in text.splitlines()]

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


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

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

        # stdout and stderr are capped independently, so a huge stdout can
        # never push a short but important stderr message out of view.
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
    """Read file contents, windowed by line number."""

    name = "read_file"
    description = (
        "Read the contents of a file, returned as a windowed slice of "
        "lines. Use offset/limit to page through large files instead of "
        "re-reading from the start — the result tells you if more is "
        "available and what offset to use next."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read",
            },
            "offset": {
                "type": "integer",
                "description": "1-indexed line number to start reading from (default: 1)",
                "default": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to return (default: 2000)",
                "default": 2000,
            },
        },
        "required": ["path"],
    }
    requires_permission = True  # Permission managed by path (auto-allow within cwd)

    def execute(self, path: str, offset: int = 1, limit: int = 2000) -> str:
        try:
            p = Path(path).expanduser()
            if not p.exists():
                return f"[error: file not found: {path}]"
            if not p.is_file():
                return f"[error: not a file: {path}]"

            lines = p.read_text().splitlines()
            total = len(lines)

            if total == 0:
                return "(empty file)"

            start = max(offset - 1, 0)
            if start >= total:
                return f"[error: offset {offset} is past end of file ({total} lines total)]"

            end = min(start + limit, total)
            chunk = [_clip_line(ln) for ln in lines[start:end]]

            # Line numbers let the model navigate and cross-reference grep
            # hits (auth.py:340 -> read_file(offset=320)) without guesswork.
            numbered = "\n".join(f"{start + i + 1}\t{ln}" for i, ln in enumerate(chunk))
            header = f"[lines {start + 1}-{end} of {total}]\n"
            if end < total:
                header += f"[more available — call again with offset={end + 1}]\n"

            return header + numbered
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
    """Search for patterns in files using ripgrep, windowed by match count."""

    name = "grep"
    description = (
        "Search for a regex pattern in files using grep. Reports total "
        "match/file counts up front, then returns a windowed slice of "
        "matches (file:line:content). Use offset/limit to page through "
        "results, or narrow the pattern/glob if the total count is large "
        "rather than paging through everything."
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
            "offset": {
                "type": "integer",
                "description": "Index of the first match to show (0-indexed, default: 0)",
                "default": 0,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of matches to show (default: 50)",
                "default": 50,
            },
        },
        "required": ["pattern"],
    }
    requires_permission = False  # Permission managed by path (auto-allow within cwd)

    # Internal safety net, independent of the user-facing `limit`: how many
    # match lines we're willing to parse from rg's raw output before giving
    # up on an exact count (protects memory on a pathological pattern that
    # matches most lines of a huge repo).
    _PARSE_CEILING = 20_000

    def execute(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        ignore_case: bool = False,
        offset: int = 0,
        limit: int = 50,
    ) -> str:
        try:
            cmd = ["rg", "--line-number", "--max-count", "2000"]  # per-file safety cap
            if ignore_case:
                cmd.append("--ignore-case")
            if glob:
                cmd.extend(["--glob", glob])
            cmd.extend([pattern, path])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 1:  # no matches
                return "(no matches)"
            if result.returncode != 0:
                return f"[error: {result.stderr}]"

            raw_lines = result.stdout.splitlines()
            ceiling_hit = len(raw_lines) > self._PARSE_CEILING
            raw_lines = raw_lines[: self._PARSE_CEILING]

            matches = []       # (file, line_no, content)
            files_seen = []    # preserve first-seen order
            files_set = set()
            for raw in raw_lines:
                parts = raw.split(":", 2)
                if len(parts) != 3:
                    continue  # skip malformed/separator lines defensively
                file_, line_no, content = parts
                matches.append((file_, line_no, content))
                if file_ not in files_set:
                    files_set.add(file_)
                    files_seen.append(file_)

            total_matches = len(matches)
            total_files = len(files_seen)

            if total_matches == 0:
                return "(no matches)"

            window = matches[offset : offset + limit]
            shown_lines = "\n".join(
                f"{f}:{ln}:{_clip_line(c)}" for f, ln, c in window
            )

            # Scope first, content second: the model can decide to narrow
            # the search before reading a wall of hits.
            approx = "~" if ceiling_hit else ""
            header = f"{approx}{total_matches} match(es) across {total_files} file(s)"
            if total_matches > limit or offset > 0:
                shown_end = offset + len(window)
                header += f" — showing {offset + 1}-{shown_end}"
                if shown_end < total_matches:
                    header += f" (call again with offset={shown_end} for more,"
                    header += " or narrow pattern/glob to reduce the count)"
            if ceiling_hit:
                header += (
                    f"\n[counts are approximate — stopped parsing after "
                    f"{self._PARSE_CEILING} raw match lines; narrow the "
                    f"search for an exact count]"
                )

            return f"{header}\n\n{shown_lines}"

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