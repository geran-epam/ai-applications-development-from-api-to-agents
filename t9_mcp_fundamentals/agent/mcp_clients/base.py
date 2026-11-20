from abc import abstractmethod, ABC
from typing import Optional, Any

from mcp import Client
from mcp.types import CallToolResult, TextContent, GetPromptResult, ReadResourceResult, Resource, TextResourceContents, BlobResourceContents, Prompt
from pydantic import AnyUrl


class MCPClient(ABC):

    def __init__(self) -> None:
        self.client: Optional[Client] = None

    @abstractmethod
    async def __aenter__(self):
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...

    async def get_tools(self) -> list[dict[str, Any]]:
        """Get available tools from MCP server"""
        if not self.client:
            raise RuntimeError("MCP client not connected. Call connect() first.")
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            }
            for tool in (await self.client.list_tools()).tools
        ]

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        """Call a specific tool on the MCP server"""
        if not self.client:
            raise RuntimeError("MCP client not connected. Call connect() first.")

        tool_result: CallToolResult = await self.client.call_tool(tool_name, tool_args)
        content = tool_result.content[0]
        print(f"    ⚙️: {content}\n")
        if isinstance(content, TextContent):
            return content.text
        return content

    async def get_resources(self) -> list[Resource]:
        """Get available resources from MCP server"""
        if not self.client:
            raise RuntimeError("MCP client not connected.")
        try:
            return (await self.client.list_resources()).resources
        except Exception as e:
            print(f"Error getting resources: {e}")
            return []

    async def get_resource(self, uri: AnyUrl) -> str:
        """Get specific resource content"""
        if not self.client:
            raise RuntimeError("MCP client not connected.")

        result = await self.client.read_resource(str(uri))
        content = result.contents[0]
        if isinstance(content, TextResourceContents):
            return content.text
        if isinstance(content, BlobResourceContents):
            return content.blob
        return str(content)

    async def get_prompts(self) -> list[Prompt]:
        """Get available prompts from MCP server"""
        if not self.client:
            raise RuntimeError("MCP client not connected.")
        try:
            return (await self.client.list_prompts()).prompts
        except Exception as e:
            print(f"Error getting prompts: {e}")
            return []

    async def get_prompt(self, name: str) -> str:
        """Get specific prompt content"""
        if not self.client:
            raise RuntimeError("MCP client not connected.")
        prompt: GetPromptResult = await self.client.get_prompt(name)
        combined_content = ""
        for message in prompt.messages:
            if hasattr(message, "content"):
                if isinstance(message.content, TextContent):
                    combined_content += message.content.text + "\n"
                elif isinstance(message.content, str):
                    combined_content += message.content + "\n"
        return combined_content