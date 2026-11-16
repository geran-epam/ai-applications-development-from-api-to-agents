import asyncio
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import json

from openai import OpenAI
from pydantic import BaseModel, Field, SecretStr

from commons.constants import OPENAI_API_KEY, OPENAI_EMBEDDINGS_MODEL, OPENAI_HOST, OPENAI_LUNA_MODEL
from t6_grounding.user_service_client import UserServiceClient

#TODO: Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)


SYSTEM_PROMPT = """You are a hobbies searching wizard.

The user message contains two sections:
- `## RAG CONTEXT:` user records retrieved from the vector store, each with `id` and `about_me`
- `## USER QUESTION:` the user's actual question

Instructions:
- Perform named entity extraction: find the hobbies in `about_me` that match the user question.
- Group the matching users by hobby. Use short lowercase hobby names (e.g. "hiking", "rock climbing").
- Return ONLY user IDs that are present in the RAG CONTEXT; never invent IDs and never return personal data.
- A user may appear under several hobbies. If nobody matches, return an empty list."""

USER_PROMPT = """## RAG CONTEXT:
{context}

## USER QUESTION:
{query}"""


class HobbyGroup(BaseModel):
    hobby: str = Field(description="Hobby name, lowercase. Sample: hiking")
    user_ids: list[int] = Field(description="IDs of users that have this hobby")


class HobbiesResponse(BaseModel):
    hobbies: list[HobbyGroup] = Field(description="Users grouped by hobby", default_factory=list)


class HobbiesWizard:

    def __init__(self, embeddings: OpenAIEmbeddings):
        self.llm_client = OpenAI(api_key=OPENAI_API_KEY, base_url=f"{OPENAI_HOST}/v1")
        self.user_client = UserServiceClient()
        # In-memory Chroma collection; Document ids == user ids
        self.vectorstore = Chroma(collection_name="user_hobbies", embedding_function=embeddings)

    async def cold_start(self):
        """Embed all current users (id + about_me only) in batches of 100."""
        print("🔎 Loading all users...")
        users = self.user_client.get_all_users()
        await self._add_users(users)
        print(f"✅ Vectorstore is ready ({len(users)} users).")

    async def _add_users(self, users: list[dict[str, Any]], batch_size: int = 100):
        documents = [
            Document(id=str(u["id"]), page_content=f"id: {u['id']}\nabout_me: {u.get('about_me', '')}")
            for u in users
        ]
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        results = await asyncio.gather(
            *[self.vectorstore.aadd_documents(batch, ids=[d.id for d in batch]) for batch in batches],
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                print(f"Batch failed: {r}")

    async def sync_vectorstore(self):
        """Compare current users with the stored ones: delete removed users, add new ones."""
        current_users = {str(u["id"]): u for u in self.user_client.get_all_users()}
        stored_ids = set(self.vectorstore.get(include=[])["ids"])

        deleted = stored_ids - current_users.keys()
        added = current_users.keys() - stored_ids
        if deleted:
            self.vectorstore.delete(ids=list(deleted))
        if added:
            await self._add_users([current_users[i] for i in added])
        print(f"🔄 Vectorstore synced: +{len(added)} new, -{len(deleted)} deleted users")

    async def retrieve_context(self, query: str, k: int = 50, score: float = 0.2) -> str:
        await self.sync_vectorstore()
        print("Retrieving context...")
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k, score_threshold=score)
        parts = []
        for doc, relevance in results:
            parts.append(doc.page_content)
            print(f"Retrieved (Score: {relevance:.3f}): {doc.id} {doc.page_content[:80]!r}")
        print("=" * 100 + "\n")
        return "\n\n".join(parts)

    def generate(self, query: str, context: str) -> HobbiesResponse:
        response = self.llm_client.beta.chat.completions.parse(
            model=OPENAI_LUNA_MODEL,
            temperature=0.0,
            reasoning_effort="none",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(context=context, query=query)},
            ],
            response_format=HobbiesResponse,
        )
        return response.choices[0].message.parsed or HobbiesResponse()

    async def ground_output(self, generated: HobbiesResponse) -> dict[str, list[dict[str, Any]]]:
        """Output grounding: fetch full user info; drop IDs that don't exist (hallucinations)."""
        all_ids = {uid for g in generated.hobbies for uid in g.user_ids}
        ids = list(all_ids)
        fetched = await asyncio.gather(*[self.user_client.get_user(i) for i in ids], return_exceptions=True)
        users: dict[int, Optional[dict[str, Any]]] = {
            i: (None if isinstance(r, Exception) else r) for i, r in zip(ids, fetched)
        }
        for i, u in users.items():
            if u is None:
                print(f"⚠️ User {i} not found - skipped")

        result: dict[str, list[dict[str, Any]]] = {}
        for group in generated.hobbies:
            found = [users[i] for i in dict.fromkeys(group.user_ids) if users.get(i)]
            if found:
                result.setdefault(group.hobby, []).extend(found)
        return result


async def main():
    embeddings = OpenAIEmbeddings(
        model=OPENAI_EMBEDDINGS_MODEL,
        api_key=SecretStr(OPENAI_API_KEY),
        base_url=f"{OPENAI_HOST}/v1",
        # OpenRouter expects raw text, not pre-tokenized input
        check_embedding_ctx_length=False,
        dimensions=384,
    )
    wizard = HobbiesWizard(embeddings)
    await wizard.cold_start()

    print("Query samples:")
    print(" - I need people who love to go to mountains")
    print(" - Who likes cooking and baking?")
    while True:
        user_question = input("> ").strip()
        if not user_question:
            continue
        if user_question.lower() in ['quit', 'exit']:
            break

        context = await wizard.retrieve_context(user_question)
        generated = wizard.generate(user_question, context)
        print(f"LLM grouping (IDs only): {generated.model_dump_json()}")
        grounded = await wizard.ground_output(generated)
        print("\n" + json.dumps(grounded, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
