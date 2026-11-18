"""MCP Client for KEHAI."""

import asyncio
import os
from contextlib import AsyncExitStack
from typing import Dict, Any, Iterable, List, Callable, Optional
from loguru import logger
from datetime import timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.types import Tool
from mcp.client.stdio import stdio_client

from .server_registry import ServerRegistry

DEFAULT_TIMEOUT = timedelta(seconds=30)
# aclose() gives each server task this long to wind down on its own (stdin
# close -> process.wait, with the SDK's 2 s SIGTERM/SIGKILL escalation on top)
# before cancelling it outright.
CLOSE_TIMEOUT = 10.0


class MCPClient:
    """MCP Client for KEHAI.
    Manages persistent connections to multiple MCP servers.

    Every server connection lives in its OWN asyncio task (`_serve`): the stdio
    transport and the ClientSession are entered there and exited there. Both
    are built on anyio task groups / cancel scopes, which refuse to be exited
    from any task but the one that entered them. The previous shared
    AsyncExitStack was entered from whichever conversation task first needed a
    server and exited from the disconnect handler, which raised "Attempted to
    exit cancel scope in a different task than it was entered in"
    (logs/server.log, 2026-10-22 16:14:22) and left the child process behind.
    Callers only ever receive the session object; the owning task parks on
    `_stop` until aclose().

    Sessions can be started ahead of the first tool call with start_servers()
    (the uvx `time` server took ~480 ms to spawn on the first call of a turn,
    all of it silence). A pre-spawn that fails is retried lazily by the next
    request, exactly as before.
    """

    def __init__(
        self,
        server_registery: ServerRegistry,
        send_text: Callable = None,
        client_uid: str = None,
    ) -> None:
        """Initialize the MCP Client."""
        self.active_sessions: Dict[str, ClientSession] = {}
        self._list_tools_cache: Dict[str, List[Tool]] = {}  # Cache for list_tools
        self._send_text: Callable = send_text
        self._client_uid: str = client_uid
        # Per-server lifecycle task, and the future that resolves to its session
        # (or to the connect error). A future that is done but whose session is
        # gone is simply replaced by the next request.
        self._server_tasks: Dict[str, asyncio.Task] = {}
        self._server_ready: Dict[str, asyncio.Future] = {}
        self._stop = asyncio.Event()
        self._closing = False

        if isinstance(server_registery, ServerRegistry):
            self.server_registery = server_registery
        else:
            raise TypeError(
                "MCPC: Invalid server manager. Must be an instance of ServerRegistry."
            )
        logger.info("MCPC: Initialized MCPClient instance.")

    def start_servers(self, server_names: Iterable[str]) -> None:
        """Pre-spawn: start the connection task for each named server NOW, in
        the background, so the first tool call of the connection finds a live
        session instead of paying the spawn. Best effort: a failure is logged by
        the task and the next request falls back to the lazy spawn in
        _ensure_server_running_and_get_session. Unknown names are skipped (the
        ToolAdapter already warned about them)."""
        started = []
        for name in server_names:
            if name in self._server_tasks or not self.server_registery.get_server(name):
                continue
            self._spawn(name)
            started.append(name)
        if started:
            logger.info(f"MCPC: Pre-spawning server(s) in the background: {started}")

    def _spawn(self, server_name: str) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        ready: asyncio.Future = loop.create_future()
        self._server_ready[server_name] = ready
        self._server_tasks[server_name] = loop.create_task(
            self._serve(server_name, ready), name=f"mcp-server:{server_name}"
        )
        return ready

    async def _serve(self, server_name: str, ready: asyncio.Future) -> None:
        """Own one server connection for its whole life, in this task."""
        server = self.server_registery.get_server(server_name)
        if not server:
            ready.set_exception(
                ValueError(
                    f"MCPC: Server '{server_name}' not found in available servers."
                )
            )
            return

        timeout = server.timeout if server.timeout else DEFAULT_TIMEOUT

        # env=None makes the MCP SDK launch the child with a MINIMAL, sanitised
        # environment. `uvx` needs the real one (USERPROFILE/APPDATA/LOCALAPPDATA
        # for its package cache on Windows, plus proxy vars), and without it a
        # server that has to resolve a dependency tree — ddg-search — died
        # instantly with "Connection closed" while the tiny `time` server, whose
        # deps were already resolved, still worked. That silently left the
        # avatar with NO web search: "GitHubのランキングを聞いても答えられない".
        # Explicit per-server `env` entries still win over the inherited ones.
        server_params = StdioServerParameters(
            command=server.command,
            args=server.args,
            env={**os.environ, **(server.env or {})},
            cwd=server.cwd,
        )

        logger.info(f"MCPC: Starting and connecting to server '{server_name}'...")
        session: Optional[ClientSession] = None
        try:
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(
                    stdio_client(server_params)
                )
                session = await stack.enter_async_context(
                    ClientSession(read, write, read_timeout_seconds=timeout)
                )
                await session.initialize()
                self.active_sessions[server_name] = session
                if not ready.done():
                    ready.set_result(session)
                logger.info(f"MCPC: Successfully connected to server '{server_name}'.")
                await self._stop.wait()
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            if not ready.done():
                ready.set_exception(
                    RuntimeError(
                        f"MCPC: Connection to server '{server_name}' was cancelled."
                    )
                )
            raise
        except BaseException as e:  # noqa: BLE001 — anyio teardown raises BaseExceptionGroup
            if not ready.done():
                logger.warning(
                    f"MCPC: Failed to connect to server '{server_name}': {e!r}"
                )
                ready.set_exception(
                    RuntimeError(f"MCPC: Failed to connect to server '{server_name}'.")
                )
            else:
                logger.warning(
                    f"MCPC: Connection to server '{server_name}' ended: {e!r}"
                )
        finally:
            if session is not None and self.active_sessions.get(server_name) is session:
                del self.active_sessions[server_name]

    async def _ensure_server_running_and_get_session(
        self, server_name: str
    ) -> ClientSession:
        """Gets the existing session or creates a new one."""
        session = self.active_sessions.get(server_name)
        if session is not None:
            return session
        if self._closing:
            raise RuntimeError(
                f"MCPC: Client is closing; cannot connect to server '{server_name}'."
            )
        if not self.server_registery.get_server(server_name):
            raise ValueError(
                f"MCPC: Server '{server_name}' not found in available servers."
            )

        ready = self._server_ready.get(server_name)
        if ready is None or ready.done():
            # Nothing in flight: no pre-spawn, the pre-spawn failed, or the
            # session went away since (its task popped active_sessions). Lazy
            # spawn, as before.
            ready = self._spawn(server_name)
        else:
            logger.info(f"MCPC: Waiting for pre-spawned server '{server_name}'...")
        # shield: a cancelled caller (barge-in) must not cancel the shared future
        return await asyncio.shield(ready)

    async def list_tools(self, server_name: str) -> List[Tool]:
        """List all available tools on the specified server."""
        # Check cache first
        if server_name in self._list_tools_cache:
            logger.debug(f"MCPC: Cache hit for list_tools on server '{server_name}'.")
            return self._list_tools_cache[server_name]

        logger.debug(
            f"MCPC: Cache miss for list_tools on server '{server_name}'. Fetching..."
        )
        session = await self._ensure_server_running_and_get_session(server_name)
        response = await session.list_tools()

        # Store in cache before returning
        self._list_tools_cache[server_name] = response.tools
        logger.debug(f"MCPC: Cached list_tools result for server '{server_name}'.")
        return response.tools

    async def call_tool(
        self, server_name: str, tool_name: str, tool_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a tool on the specified server.

        Returns:
            Dict containing the metadata and content_items from the tool response.
        """
        session = await self._ensure_server_running_and_get_session(server_name)
        logger.info(f"MCPC: Calling tool '{tool_name}' on server '{server_name}'...")
        response = await session.call_tool(tool_name, tool_args)

        if response.isError:
            error_text = (
                response.content[0].text
                if response.content and hasattr(response.content[0], "text")
                else "Unknown server error"
            )
            logger.error(f"MCPC: Error calling tool '{tool_name}': {error_text}")
            # Return error information within the standard structure
            return {
                "metadata": getattr(response, "metadata", {}),
                "content_items": [{"type": "error", "text": error_text}],
            }

        content_items = []
        if response.content:
            for item in response.content:
                item_dict = {"type": getattr(item, "type", "text")}
                # Extract available attributes from content item
                for attr in [
                    "text",
                    "data",
                    "mimeType",
                    "url",
                    "altText",
                ]:  # Added url and altText
                    if (
                        hasattr(item, attr) and getattr(item, attr) is not None
                    ):  # Check for None
                        item_dict[attr] = getattr(item, attr)
                content_items.append(item_dict)
        else:
            logger.warning(
                f"MCPC: Tool '{tool_name}' returned no content. Returning empty content_items."
            )
            content_items.append(
                {"type": "text", "text": ""}
            )  # Ensure content_items is not empty

        result = {
            "metadata": getattr(response, "metadata", {}),
            "content_items": content_items,
        }
        return result

    async def aclose(self) -> None:
        """Closes all active server connections, each in the task that owns it."""
        tasks = [t for t in self._server_tasks.values() if not t.done()]
        logger.info(
            f"MCPC: Closing client instance and {len(self.active_sessions)} active connections..."
        )
        self._closing = True
        self._stop.set()
        try:
            if tasks:
                _done, pending = await asyncio.wait(tasks, timeout=CLOSE_TIMEOUT)
                for task in pending:
                    logger.warning(
                        f"MCPC: {task.get_name()} did not stop within "
                        f"{CLOSE_TIMEOUT}s; cancelling."
                    )
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
        finally:
            self._server_tasks.clear()
            self._server_ready.clear()
            self.active_sessions.clear()
            self._list_tools_cache.clear()  # Clear cache on close
            # Re-armed, so the instance stays reusable (as the old exit stack was).
            self._stop = asyncio.Event()
            self._closing = False
        logger.info("MCPC: Client instance closed.")

    async def __aenter__(self) -> "MCPClient":
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the async context manager."""
        await self.aclose()
        if exc_type:
            logger.error(f"MCPC: Exception in async context: {exc_value}")


# if __name__ == "__main__":
#     # Test the MCPClient.
#     async def main():
#         server_registery = ServerRegistry()
#         async with MCPClient(server_registery) as client:
#             # Assuming 'example' server and 'example_tool' exist
#             # The old call used: await client.call_tool("example_tool", {"arg1": "value1"})
#             # The new call needs server name:
#             try:
#                 result = await client.call_tool("example", "example_tool", {"arg1": "value1"})
#                 print(f"Tool result: {result}")
#                 # Test error handling by calling a non-existent tool
#                 await client.call_tool("example", "non_existent_tool", {})
#             except ValueError as e:
#                 print(f"Caught expected error: {e}")
#             except Exception as e:
#                 print(f"Caught unexpected error: {e}")

#     asyncio.run(main())
