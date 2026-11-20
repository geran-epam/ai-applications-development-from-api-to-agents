import os
import sys
import asyncio
from pathlib import Path

from commons.constants import OPENAI_API_KEY, OPENAI_TERRA_MODEL
from commons.models.message import Message
from commons.models.role import Role
from t9_mcp_fundamentals.agent.agent import AgentMCPFundamentals
from t9_mcp_fundamentals.agent.mcp_clients.base import MCPClient
from t9_mcp_fundamentals.agent.mcp_clients.http import HttpMCPClient
from t9_mcp_fundamentals.agent.mcp_clients.stdio import StdioMCPClient
from t9_mcp_fundamentals.agent.prompts import SYSTEM_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STDIO_SERVER_PATH = PROJECT_ROOT / "t9_mcp_fundamentals" / "mcp_server" / "stdio_server.py"


def create_client() -> MCPClient:
    """Select MCP client by MCP_MODE: http (default), ddg_http, stdio, ddg_stdio."""
    mode = os.getenv("MCP_MODE", "http")
    if mode == "http":
        return HttpMCPClient(mcp_server_url="http://localhost:8005/mcp")
    if mode == "ddg_http":
        return HttpMCPClient(mcp_server_url="http://localhost:8010/mcp")
    if mode == "stdio":
        return StdioMCPClient(
            command=sys.executable,
            args=[str(STDIO_SERVER_PATH)],
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        )
    if mode == "ddg_stdio":
        return StdioMCPClient(docker_image="mcp/duckduckgo:latest")
    raise ValueError(f"Unknown MCP_MODE: {mode}")


async def main():
    async with create_client() as mcp_client:
        print("\nAvailable Resources:")
        for resource in await mcp_client.get_resources():
            print(f"  - {resource.uri} ({resource.name}, {resource.mime_type}): {resource.description}")

        tools = await mcp_client.get_tools()
        print("\nAvailable Tools:")
        for tool in tools:
            print(f"  - {tool['function']['name']}: {tool['function']['description']}")

        agent = AgentMCPFundamentals(api_key=OPENAI_API_KEY, model=OPENAI_TERRA_MODEL, tools=tools,
                                     mcp_client=mcp_client)
        messages: list[Message] = [Message(role=Role.SYSTEM, content=SYSTEM_PROMPT)]

        print("\nAvailable Prompts:")
        for prompt in await mcp_client.get_prompts():
            print(f"  - {prompt.name}: {prompt.description}")

        while True:
            user_input = input("\n> ").strip()
            if user_input.lower() == 'exit':
                break
            if not user_input:
                continue
            # echo, since piped stdin is not shown in the captured output
            print(f"[user input] {user_input}")
            messages.append(Message(role=Role.USER, content=user_input))
            ai_message = await agent.get_response(messages)
            messages.append(ai_message)


if __name__ == "__main__":
    asyncio.run(main())
