from typing import Any

from t8_agent.task.tools.users.base import BaseUserServiceTool


class SearchUsersTool(BaseUserServiceTool):
    MAX_CHARS = 6000

    @property
    def name(self) -> str:
        return "search_users"

    @property
    def description(self) -> str:
        return (
            "Searches users by name, surname, email or gender. "
            "All parameters are optional; provided ones are combined as filters."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the user"},
                "surname": {"type": "string", "description": "Surname of the user"},
                "email": {"type": "string", "description": "Email of the user"},
                "gender": {"type": "string", "description": "Gender of the user"},
            },
            "required": [],
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        try:
            result = self._user_client.search_users(**arguments)
            if len(result) > self.MAX_CHARS:
                return result[:self.MAX_CHARS] + f"\n... [truncated, output exceeded {self.MAX_CHARS} characters; refine the search filters]"
            return result
        except Exception as e:
            return f"Error while searching users: {str(e)}"
