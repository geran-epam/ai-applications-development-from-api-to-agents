import os

from commons.constants import OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENAI_TERRA_MODEL, ANTHROPIC_SONNET_MODEL
from commons.models.conversation import Conversation
from commons.models.message import Message
from commons.models.role import Role
from commons.user_service.client import UserServiceClient

from t8_agent.task.agents.anthropic import AnthropicBasedAgent
from t8_agent.task.agents.openai import OpenAIBasedAgent
from t8_agent.task.prompts import SYSTEM_PROMPT
from t8_agent.task.tools.users.create_user_tool import CreateUserTool
from t8_agent.task.tools.users.delete_user_tool import DeleteUserTool
from t8_agent.task.tools.users.get_user_by_id_tool import GetUserByIdTool
from t8_agent.task.tools.users.search_users_tool import SearchUsersTool
from t8_agent.task.tools.users.update_user_tool import UpdateUserTool
from t8_agent.task.tools.web_search import WebSearchTool


def main():
    user_client = UserServiceClient()
    tools = [
        WebSearchTool(OPENAI_API_KEY),
        GetUserByIdTool(user_client),
        SearchUsersTool(user_client),
        CreateUserTool(user_client),
        UpdateUserTool(user_client),
        DeleteUserTool(user_client),
    ]

    # Provider is selected with AGENT_PROVIDER=openai|anthropic (default: openai)
    if os.getenv("AGENT_PROVIDER", "openai").lower() == "anthropic":
        agent = AnthropicBasedAgent(ANTHROPIC_SONNET_MODEL, ANTHROPIC_API_KEY, tools, SYSTEM_PROMPT)
    else:
        agent = OpenAIBasedAgent(OPENAI_TERRA_MODEL, OPENAI_API_KEY, tools, SYSTEM_PROMPT)

    conversation = Conversation()
    while True:
        try:
            user_input = input("> ").strip()
        except EOFError:
            break
        if user_input.lower() == "exit":
            break
        if not user_input:
            continue

        conversation.add_message(Message(role=Role.USER, content=user_input))
        ai_message = agent.get_response(conversation.get_messages())
        conversation.add_message(ai_message)
        print(f"\nAI: {ai_message.content}\n")


main()
