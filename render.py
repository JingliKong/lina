"""Rendering helpers for tool call previews."""

import difflib
from pathlib import Path

from rich.console import Group, RenderableType
from rich.syntax import Syntax
from rich.text import Text
from rich.markup import escape as rich_escape

from messages import ToolCall

DIFF_ADD = "#3fb950 on #0f2417"
DIFF_DEL = "#f85149 on #2a1315"
DIFF_CTX = "#8b949e"
DIFF_GUTTER = "#484f58"
DIFF_HUNK = "#a371f7"

# Arg names whose values are file bodies rather than scalars.
_BODY_KEYS = ("content", "file_text", "text", "new_str", "new_string", "old_str", "old_string")

# Tool names that mutate a file and should be previewed as a diff.
_EDIT_TOOLS = ("write_file", "edit_file", "str_replace")

# Accepted spellings for each argument, most preferred first.
_PATH_KEYS = ("path", "file_path")
_NEW_FILE_KEYS = ("content", "file_text")
_OLD_STR_KEYS = ("old_string", "old_str")
_NEW_STR_KEYS = ("new_string", "new_str")


def _first(args: dict, keys) -> str | None:
    """Return the first present, non-None value among `keys`."""
    for key in keys:
        if args.get(key) is not None:
            return args[key]
    return None


def file_text(path: str) -> str | None:
    try:
        return Path(path).expanduser().read_text()
    except (OSError, UnicodeDecodeError):
        return None


def lang_for(path: str) -> str:
    return "python" if path.endswith(".py") else "text"


def preview(code: str, lang: str, max_lines: int | None = 20) -> Syntax:
    lines = code.splitlines()
    if max_lines is not None and len(lines) > max_lines:
        code = "\n".join(lines[:max_lines]) + f"\n… ({len(lines) - max_lines} more lines)"
    return Syntax(code, lang, theme="ansi_dark", word_wrap=True)


def _diff_line(old_no, new_no, marker: str, text: str, style: str) -> Text:
    gutter = f"{old_no or '':>4} {new_no or '':>4} "
    line = Text.assemble((gutter, DIFF_GUTTER), (f"{marker} {text}", style))
    line.no_wrap = True
    line.overflow = "ellipsis"
    return line


def render_diff(old_text: str, new_text: str,
                context: int = 3, max_lines: int | None = 40) -> list[Text]:
    """Render a colored, gutter-numbered diff.

    Pass max_lines=None to render every changed line with no truncation.
    """
    a, b = old_text.splitlines(), new_text.splitlines()
    groups = list(difflib.SequenceMatcher(None, a, b).get_grouped_opcodes(context))
    if not groups:
        return [Text("(no changes)", style="dim italic")]

    out: list[Text] = []
    for group in groups:
        i_start, j_start = group[0][1], group[0][3]
        i_end, j_end = group[-1][2], group[-1][4]
        out.append(Text(
            f"@@ -{i_start + 1},{i_end - i_start} +{j_start + 1},{j_end - j_start} @@",
            style=DIFF_HUNK,
        ))
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                for k, line in enumerate(a[i1:i2]):
                    out.append(_diff_line(i1 + k + 1, j1 + k + 1, " ", line, DIFF_CTX))
                continue
            if tag in ("replace", "delete"):
                for k, line in enumerate(a[i1:i2]):
                    out.append(_diff_line(i1 + k + 1, None, "-", line, DIFF_DEL))
            if tag in ("replace", "insert"):
                for k, line in enumerate(b[j1:j2]):
                    out.append(_diff_line(None, j1 + k + 1, "+", line, DIFF_ADD))

    if max_lines is not None and len(out) > max_lines:
        out = out[:max_lines] + [Text(f"… {len(out) - max_lines} more diff lines", style="dim italic")]
    return out


def diff_stat(old_text: str, new_text: str) -> Text:
    """One-line +added / -removed summary."""
    a, b = old_text.splitlines(), new_text.splitlines()
    added = removed = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    return Text.assemble(
        (f"+{added}", DIFF_ADD.split(" on ")[0]),
        ("  ", ""),
        (f"-{removed}", DIFF_DEL.split(" on ")[0]),
    )


def action_label(call: ToolCall) -> str:
    args = call.args
    if _first(args, _NEW_FILE_KEYS) is not None:
        return "Write to"
    if _first(args, _OLD_STR_KEYS) is not None or _first(args, _NEW_STR_KEYS) is not None:
        return "Edit"
    if call.name == "read_file":
        return "Read"
    return "Path"


def _edited_text(args: dict, old: str) -> str | None:
    """Reconstruct the post-edit file body from a tool call's args."""
    whole_file = _first(args, _NEW_FILE_KEYS)
    if whole_file is not None:
        return whole_file

    old_s = _first(args, _OLD_STR_KEYS)
    new_s = _first(args, _NEW_STR_KEYS)
    if old_s is not None and new_s is not None:
        if old_s not in old:
            return None  # edit won't apply; fall back to raw arg rendering
        return old.replace(old_s, new_s, 1)
    return None


def render_args(call: ToolCall) -> RenderableType:
    """Render a tool call's arguments in full — no truncation."""
    args = dict(call.args)

    if call.name in _EDIT_TOOLS:
        path = _first(args, _PATH_KEYS)
        if path:
            old = file_text(path) or ""
            new = _edited_text(args, old)
            if new is not None:
                header = Text.assemble(
                    (f"{action_label(call)} ", "bold"),
                    (str(path), "bold cyan"),
                    ("  ", ""),
                )
                header.append_text(diff_stat(old, new))
                return Group(
                    header,
                    Text(""),
                    *render_diff(old, new, max_lines=None),
                )

    parts: list[RenderableType] = []
    for key, value in args.items():
        if key in _BODY_KEYS and isinstance(value, str) and "\n" in value:
            parts.append(Text(f"{key}:", style="bold"))
            parts.append(
                Syntax(
                    value,
                    lang_for(str(_first(args, _PATH_KEYS) or "")),
                    theme="ansi_dark",
                    word_wrap=True,
                )
            )
        else:
            parts.append(Text.assemble((f"{key}: ", "bold"), rich_escape(str(value))))
    return Group(*parts)