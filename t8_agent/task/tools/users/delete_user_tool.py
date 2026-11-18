from typing import Any

from t8_agent.task.tools.users.base import BaseUserServiceTool

class DeleteUserTool(BaseUserServiceTool):

    @property
    def name(self) -> str:
        return "delete_users"

    @property
    def description(self) -> str:
        return "Deletes a user by ID. This action is irreversible."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "number", "description": "ID of the user that should be deleted"}
            },
            "required": ["id"],
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        try:
            return self._user_client.delete_user(int(arguments["id"]))
        except Exception as e:
            return f"Error while deleting user by id: {str(e)}"
