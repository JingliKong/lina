"""GitHub Copilot SDK provider.

Unlike the other providers, the Copilot SDK is not a stateless chat-completions
API — it is an agent runtime (the Copilot CLI in server mode, driven over
JSON-RPC). It owns the conversation history and it owns the tool-calling loop.

To fit Henri's Provider contract we run in "bridge" mode:

  * one long-lived CopilotSession per provider instance holds the history, so
    ``stream()`` only sends the newest user message rather than replaying
    everything;
  * Henri's tools are registered as Copilot *custom tools* whose handlers do
    not execute anything. They park on an asyncio.Future and hand the call
    back to Henri as a normal ``StreamEvent(tool_calls=...)``;
  * the next ``stream()`` call carries the tool results, which resolve those
    futures and let the Copilot turn continue.

Net effect: Henri still decides what runs, and the Copilot agent loop is what
drives the model.
"""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
import asyncio

from copilot import CopilotClient, PermissionHandler, ToolSet
from copilot import Tool as CopilotTool
from copilot import ToolInvocation, ToolResult
from copilot.session import CopilotSession

if TYPE_CHECKING:
    from tools.base import Tool

from config import DEFAULT_COPILOT_MODEL
from messages import Message, ToolCall
from providers.base import Provider, StreamEvent, Usage

TOOL_PREFIX = "henri_"



class CopilotProvider(Provider):
    """GitHub Copilot SDK provider (Copilot CLI agent runtime)."""

    name = "copilot"

    def __init__(
        self,
        model_id: str = DEFAULT_COPILOT_MODEL,
        working_directory: str | None = None,
        github_token: str | None = None,
        reasoning_effort: str | None = None,
        builtin_tools: bool = False,
        emit_reasoning: bool = False,
    ):
        """
        Args:
            model_id: Copilot model name ("gpt-5", "claude-sonnet-4.5", "auto", ...).
            working_directory: Session working directory.
            github_token: Explicit token; otherwise the SDK uses the logged-in
                user or COPILOT_GITHUB_TOKEN / GH_TOKEN / GITHUB_TOKEN.
            reasoning_effort: "low" | "medium" | "high" | "xhigh" | "max".
            builtin_tools: Let Copilot use its own file/shell/web tools in
                addition to Henri's. Off by default so Henri stays in control
                of every side effect.
            emit_reasoning: Stream extended-thinking deltas as text.
        """
        self.model_id = model_id
        self.working_directory = working_directory
        self.github_token = github_token
        self.reasoning_effort = reasoning_effort
        self.builtin_tools = builtin_tools
        self.emit_reasoning = emit_reasoning

        self._client: CopilotClient | None = None
        self._session: CopilotSession | None = None
        self._unsubscribe = None

        # Events pulled off the SDK's callback and fed to stream().
        self._queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        # tool_call_id -> future the parked tool handler is awaiting.
        self._pending: dict[str, asyncio.Future[str]] = {}
        # Signature of the tool set the live session was built with.
        self._session_key: tuple | None = None
        # Message ids we've already streamed as deltas (avoid double-emitting).
        self._streamed: set[str] = set()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def _copilot_tool_name(self, name: str) -> str:
        """Avoid collisions with Copilot built-in tools."""
        return f"{TOOL_PREFIX}{name}"
    
    def _tools_to_copilot(self, tools: list["Tool"]) -> list[CopilotTool]:
        """Wrap Henri tools as declaration-only Copilot custom tools."""
        copilot_tools: list[CopilotTool] = []

        for tool in tools:
            copilot_name = self._copilot_tool_name(tool.name)
            copilot_tools.append(
                CopilotTool(
                    name=copilot_name,
                    description=tool.description,
                    parameters=tool.parameters,
                    handler=self._make_handler(tool.name),
                    skip_permission=True,  # Henri approves tools, not the CLI
                )
            )

        return copilot_tools

    def _make_handler(self, name: str):
        """Build a handler that hands the call to Henri and waits for a result."""

        async def handler(invocation: ToolInvocation) -> ToolResult:
            future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
            self._pending[invocation.tool_call_id] = future
            self._queue.put_nowait((
                "tool_call",
                ToolCall(
                    id=invocation.tool_call_id,
                    name=name,
                    args=invocation.arguments or {},
                ),
            ))
            try:
                content = await future
            except asyncio.CancelledError:
                return ToolResult(
                    text_result_for_llm="Tool call cancelled by the host.",
                    result_type="error",
                )
            finally:
                self._pending.pop(invocation.tool_call_id, None)
            return ToolResult(text_result_for_llm=content, result_type="success")

        return handler

    def _on_event(self, event) -> None:
        """SDK callback (sync). Translate events onto the queue."""
        etype = event.type.value
        data = event.data

        # Sub-agent chatter would interleave badly with the main stream.
        if event.agent_id is not None:
            return

        if etype == "assistant.message_delta":
            self._streamed.add(getattr(data, "message_id", ""))
            if delta := getattr(data, "delta_content", ""):
                self._queue.put_nowait(("text", delta))
        elif etype == "assistant.reasoning_delta" and self.emit_reasoning:
            if delta := getattr(data, "delta_content", ""):
                self._queue.put_nowait(("text", delta))
        elif etype == "assistant.message":
            # Only fires as the sole text source when streaming is off.
            if getattr(data, "message_id", "") not in self._streamed:
                if content := getattr(data, "content", ""):
                    self._queue.put_nowait(("text", content))
        elif etype == "assistant.usage":
            self._queue.put_nowait((
                "usage",
                Usage(
                    input_tokens=getattr(data, "input_tokens", 0) or 0,
                    output_tokens=getattr(data, "output_tokens", 0) or 0,
                ),
            ))
        elif etype == "session.idle":
            self._queue.put_nowait(("idle", getattr(data, "aborted", False)))
        elif etype == "session.error":
            self._queue.put_nowait((
                "error",
                f"{getattr(data, 'error_type', 'error')}: {getattr(data, 'message', '')}",
            ))

    async def _ensure_session(
        self, tools: list["Tool"], system: str
    ) -> CopilotSession:
        """Start the client and (re)create the session if the setup changed."""
        if self._client is None:
            self._client = CopilotClient(
                working_directory=self.working_directory,
                github_token=self.github_token,
            )
            await self._client.start()

        key = (tuple(sorted(t.name for t in tools)), system, self.model_id)
        if self._session is not None and self._session_key == key:
            return self._session

        await self._close_session()

        available = ToolSet().add_custom("*")
        if self.builtin_tools:
            available.add_builtin("*")

        self._session = await self._client.create_session(
            model=self.model_id,
            streaming=True,
            reasoning_effort=self.reasoning_effort,
            tools=self._tools_to_copilot(tools),
            available_tools=available,
            system_message={"mode": "replace", "content": system} if system else None,
            working_directory=self.working_directory,
            on_permission_request=PermissionHandler.approve_all,
        )
        self._unsubscribe = self._session.on(self._on_event)
        self._session_key = key
        return self._session

    async def _close_session(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()
        self._streamed.clear()
        while not self._queue.empty():
            self._queue.get_nowait()
        if self._session is not None:
            await self._session.disconnect()
            self._session = None
        self._session_key = None

    async def close(self) -> None:
        """Tear down the session and the underlying CLI process."""
        await self._close_session()
        if self._client is not None:
            await self._client.stop()
            self._client = None

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _deliver_results(self, messages: list[Message]) -> bool:
        """Resolve parked tool handlers from the newest tool results.

        Returns True if this turn is a continuation (no new prompt to send).
        """
        if not self._pending or not messages:
            return False

        results = messages[-1].tool_results
        if not results:
            return False

        delivered = False
        for tr in results:
            call_id = getattr(tr, "tool_call_id", None) or getattr(tr, "id", None)
            future = self._pending.get(call_id) if call_id else None
            if future is None:
                # No usable id — fall back to oldest outstanding call.
                for pending in self._pending.values():
                    if not pending.done():
                        future = pending
                        break
            if future is not None and not future.done():
                future.set_result(tr.content or "")
                delivered = True
        return delivered

    def _latest_prompt(self, messages: list[Message]) -> str:
        for msg in reversed(messages):
            if msg.role == "user" and msg.content:
                return msg.content
        return ""

    async def stream(
        self,
        messages: list[Message],
        tools: list["Tool"],
        system: str = "",
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response from the Copilot agent runtime."""
        session = await self._ensure_session(tools, system)

        if not self._deliver_results(messages):
            prompt = self._latest_prompt(messages)
            if prompt:
                await session.send(prompt)

        usage = Usage(input_tokens=0, output_tokens=0)
        tool_calls: list[ToolCall] = []

        while True:
            kind, payload = await self._queue.get()

            if kind == "text":
                yield StreamEvent(text=payload)

            elif kind == "usage":
                usage = Usage(
                    input_tokens=usage.input_tokens + payload.input_tokens,
                    output_tokens=usage.output_tokens + payload.output_tokens,
                )

            elif kind == "tool_call":
                if not tool_calls:
                    yield StreamEvent(tool_use_started=True)
                tool_calls.append(payload)
                # Hand control back to Henri. The Copilot turn stays open with
                # the handler parked; the next stream() call resolves it.
                yield StreamEvent(
                    tool_calls=tool_calls,
                    stop_reason="tool_use",
                    usage=usage,
                )
                return

            elif kind == "idle":
                yield StreamEvent(stop_reason="end_turn", usage=usage)
                return

            elif kind == "error":
                raise RuntimeError(f"Copilot session error: {payload}")