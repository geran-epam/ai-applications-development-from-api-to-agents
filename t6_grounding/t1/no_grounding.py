import asyncio
from typing import Any

from openai import AsyncOpenAI

from commons.constants import OPENAI_API_KEY, OPENAI_HOST, OPENAI_LUNA_MODEL
from t6_grounding.user_service_client import UserServiceClient

BATCH_SYSTEM_PROMPT = """You are a user search assistant. You receive a list of users and a user question with search criteria.

Instructions:
- Analyze the search criteria from the user question.
- Examine each user in the provided list and determine whether they match the criteria.
- Return the full details of every matching user in their ORIGINAL format, without changing any value.
- If no users match, return exactly: NO_MATCHES_FOUND (nothing else).
- Use only the provided user data, never your own knowledge."""

FINAL_SYSTEM_PROMPT = """You are a user search assistant that compiles the final search results.

Instructions:
- Review all the batch search results provided.
- Combine them and deduplicate users found across several batches.
- Present the results in a clear, organized manner and keep the user data exactly as provided.
- Answer the user question using only the provided results."""

USER_PROMPT = """## USER DATA:
{context}

## USER QUESTION:
{query}"""


class TokenTracker:

    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens: list[int] = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self) -> dict:
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens,
        }


llm_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=f"{OPENAI_HOST}/v1")

token_tracker = TokenTracker()


def join_context(context: list[dict[str, Any]]) -> str:
    result = ""
    for user in context:
        result += "User:\n"
        for key, value in user.items():
            result += f"  {key}: {value}\n"
        result += "\n"
    return result


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    response = await llm_client.chat.completions.create(
        model=OPENAI_LUNA_MODEL,
        temperature=0.0,
        reasoning_effort="none",
        messages=messages,
    )
    total_tokens = response.usage.total_tokens if response.usage else 0
    token_tracker.add_tokens(total_tokens)
    content = response.choices[0].message.content or ""
    print(f"Response:\n{content}\nTokens used: {total_tokens}\n")
    return content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()

    if user_question:
        print("\n--- Searching user database ---")
        users = UserServiceClient().get_all_users()
        batches = [users[i:i + 100] for i in range(0, len(users), 100)]

        tasks = [
            generate_response(
                BATCH_SYSTEM_PROMPT,
                USER_PROMPT.format(context=join_context(batch), query=user_question),
            )
            for batch in batches
        ]
        batch_results = await asyncio.gather(*tasks)

        print("\n--- Compiling results ---")
        relevant_results = [r for r in batch_results if r.strip() != "NO_MATCHES_FOUND"]

        print("\n=== SEARCH RESULTS ===")
        if relevant_results:
            combined_results = "\n\n".join(relevant_results)
            await generate_response(
                FINAL_SYSTEM_PROMPT,
                f"## SEARCH RESULTS:\n{combined_results}\n\n## USER QUESTION:\n{user_question}",
            )
        else:
            print("No users found matching your query.")
            print("Try refining the search criteria.")

        summary = token_tracker.get_summary()
        print("\n=== Performance ===")
        print(f"Total API calls: {summary['batch_count']}")
        print(f"Total tokens: {summary['total_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())


# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation