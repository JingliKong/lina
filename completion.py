"""@-mention file completion and expansion for Henri.

Typing `@` in the prompt opens a completion menu of files in the project.
Before a message is sent to the model, `expand_file_references` replaces the
mentions with the actual file contents.

Files that shouldn't ever be pulled into a prompt (.env, keys, credentials,
build output, binaries) are filtered out of both the menu and the expansion.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import time
from pathlib import Path

from prompt_toolkit.completion import Completer, Completion

# Directories that are never worth indexing.
EXCLUDED_DIRS = {
    ".git", ".hg", ".svn", ".jj",
    "node_modules", "bower_components", "vendor",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    ".venv", "venv", "env", "site-packages", ".eggs",
    "dist", "build", "target", "out", ".next", ".nuxt", ".parcel-cache",
    ".terraform", ".gradle", ".idea", ".vscode", ".DS_Store",
}

# Filename patterns that are secret-bearing or useless as context.
EXCLUDED_GLOBS = (
    # Secrets and credentials
    ".env", ".env.*", "*.env", "env.*.local",
    ".envrc", ".netrc", ".npmrc", ".pypirc", ".htpasswd",
    "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore", "*.ppk",
    "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*",
    "*secret*", "*secrets*", "credentials", "credentials.*",
    "*.token", "*token.json", "service-account*.json",
    "*.kdbx", ".git-credentials", "*.asc", "*.gpg",
    # Lock / generated noise
    "*.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "uv.lock", "Cargo.lock", "*.min.js", "*.min.css",
    "*.map",
    # Binaries and data blobs
    "*.pyc", "*.pyo", "*.so", "*.o", "*.a", "*.dylib", "*.dll", "*.exe",
    "*.bin", "*.class", "*.jar", "*.war", "*.wasm",
    "*.zip", "*.tar", "*.gz", "*.tgz", "*.bz2", "*.xz", "*.7z", "*.rar",
    "*.db", "*.sqlite", "*.sqlite3", "*.mdb", "*.pdb",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.ico", "*.webp",
    "*.mp3", "*.mp4", "*.mov", "*.avi", "*.wav", "*.pdf",
    "*.woff", "*.woff2", "*.ttf", "*.eot", "*.otf",
    "*.log", ".DS_Store", "Thumbs.db",
)

MAX_INDEXED_FILES = 20_000
MAX_FILE_BYTES = 200_000
INDEX_TTL_SECONDS = 5.0

# `@path`, or `@"path with spaces"`.
MENTION_RE = re.compile(r'@(?:"([^"\n]+)"|([^\s"]+))')

_TRAILING_PUNCT = ".,;:!?)]}'\""


def is_excluded(rel_path: str) -> bool:
    """True if this path should never be offered or read."""
    parts = Path(rel_path).parts
    if any(part in EXCLUDED_DIRS for part in parts[:-1]):
        return True
    name = parts[-1] if parts else rel_path
    lowered = name.lower()
    return any(fnmatch.fnmatch(lowered, pattern.lower()) for pattern in EXCLUDED_GLOBS)


def _git_files(root: Path) -> list[str] | None:
    """Tracked + untracked files, honoring .gitignore. None if not a repo."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root, capture_output=True, timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return [p for p in proc.stdout.decode("utf-8", "replace").split("\0") if p]


def _walk_files(root: Path) -> list[str]:
    """Fallback index for non-git directories."""
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS and not d.startswith(".")]
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            found.append(rel.replace(os.sep, "/"))
            if len(found) >= MAX_INDEXED_FILES:
                return found
    return found


class FileIndex:
    """Cached, filtered list of project files relative to the root."""

    def __init__(self, root: str | os.PathLike | None = None):
        self.root = Path(root or Path.cwd()).resolve()
        self._cache: list[str] = []
        self._cached_at = 0.0

    def files(self) -> list[str]:
        now = time.monotonic()
        if self._cache and now - self._cached_at < INDEX_TTL_SECONDS:
            return self._cache
        raw = _git_files(self.root)
        if raw is None:
            raw = _walk_files(self.root)
        self._cache = sorted(p for p in raw[:MAX_INDEXED_FILES] if not is_excluded(p))
        self._cached_at = now
        return self._cache

    def match(self, prefix: str, limit: int = 30) -> list[str]:
        """Rank files against what the user has typed after the `@`."""
        files = self.files()
        if not prefix:
            return files[:limit]

        needle = prefix.lower()
        scored: list[tuple[tuple[int, int, int], str]] = []
        for path in files:
            low = path.lower()
            base = low.rsplit("/", 1)[-1]
            if base.startswith(needle):
                rank = 0
            elif needle in base:
                rank = 1
            elif needle in low:
                rank = 2
            elif _is_subsequence(needle, low):
                rank = 3
            else:
                continue
            scored.append(((rank, len(path), path.count("/")), path))

        scored.sort()
        return [path for _, path in scored[:limit]]


def _is_subsequence(needle: str, haystack: str) -> bool:
    it = iter(haystack)
    return all(char in it for char in needle)


class FileReferenceCompleter(Completer):
    """Completes `@...` mentions with project file paths."""

    def __init__(self, root: str | os.PathLike | None = None):
        self.index = FileIndex(root)

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        match = re.search(r'@(?:"([^"\n]*)|([^\s"]*))$', text)
        if not match:
            return
        quoted, bare = match.groups()
        prefix = quoted if quoted is not None else (bare or "")

        for path in self.index.match(prefix):
            insert = f'"{path}"' if " " in path else path
            start = -len(prefix) - (1 if quoted is not None else 0)
            yield Completion(
                insert,
                start_position=start,
                display=path,
                display_meta=_meta(self.index.root / path),
            )


def _meta(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    if path.is_dir():
        return "dir"
    for unit in ("B", "KB", "MB"):
        if size < 1024 or unit == "MB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return ""


def _resolve(raw: str, root: Path) -> tuple[str, Path] | None:
    """Resolve a mention to (mention_text, path), trimming trailing punctuation."""
    candidate = raw
    while candidate:
        path = (root / os.path.expanduser(candidate)).resolve()
        if path.exists():
            return candidate, path
        if candidate[-1] in _TRAILING_PUNCT:
            candidate = candidate[:-1]
        else:
            return None
    return None


def _read(path: Path) -> str:
    if path.is_dir():
        try:
            entries = sorted(
                p.name + ("/" if p.is_dir() else "")
                for p in path.iterdir()
                if not is_excluded(p.name)
            )
        except OSError as exc:
            return f"[error listing directory: {exc}]"
        return "\n".join(entries[:200]) or "[empty directory]"

    try:
        raw = path.read_bytes()
    except OSError as exc:
        return f"[error reading file: {exc}]"
    if b"\0" in raw[:4096]:
        return "[binary file, not included]"
    text = raw.decode("utf-8", "replace")
    if len(text) > MAX_FILE_BYTES:
        text = text[:MAX_FILE_BYTES] + "\n... [truncated]"
    return text


def expand_file_references(
    user_input: str,
    root: str | os.PathLike | None = None,
    console=None,
) -> str:
    """Append the contents of any `@`-mentioned files to the user's message."""
    root = Path(root or Path.cwd()).resolve()
    blocks: list[str] = []
    seen: set[str] = set()

    for match in MENTION_RE.finditer(user_input):
        raw = match.group(1) or match.group(2)
        resolved = _resolve(raw, root)
        if not resolved:
            continue
        mention, path = resolved
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = str(path)
        if rel in seen:
            continue
        seen.add(rel)

        if is_excluded(rel):
            if console:
                console.print(f"[yellow]Skipped {rel} (excluded from @ references)[/yellow]")
            continue

        blocks.append(f'<file path="{rel}">\n{_read(path)}\n</file>')
        if console:
            console.print(f"[dim]  @ {rel}[/dim]")

    if not blocks:
        return user_input
    return user_input + "\n\n" + "\n\n".join(blocks)