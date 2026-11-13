import os

from commons.constants import OPENAI_API_KEY, OPENAI_EMBEDDINGS_ENDPOINT, OPENAI_CHAT_COMPLETIONS_ENDPOINT, OPENAI_EMBEDDINGS_MODEL, OPENAI_TERRA_MODEL
from commons.models.conversation import Conversation
from commons.models.message import Message
from commons.models.role import Role
from t5_rag_advanced.chat.chat_completion_client import ChatCompletionClient
from t5_rag_advanced.embeddings.embeddings_client import EmbeddingsClient
from t5_rag_advanced.embeddings.text_processor import TextProcessor, SearchMode

_MANUAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings", "microwave_manual.txt")
_DB_CONFIG = {'host': 'localhost', 'port': 5433, 'database': 'vectordb', 'user': 'postgres', 'password': 'postgres'}
_DIMENSIONS = 384

SYSTEM_PROMPT = """You are a RAG-powered assistant that answers questions about microwave oven usage based on its user manual.

Each user message consists of 2 blocks:
- `##RAG CONTEXT:` information retrieved from the microwave manual based on the user's question
- `##USER QUESTION:` the user's actual question

Instructions:
- Use ONLY the information from `RAG CONTEXT` and the conversation history to answer.
- Strictly do NOT answer questions that are not related to microwave usage, not covered by `RAG CONTEXT`,
  or are out of the conversation history scope. In that case reply that you can only answer questions covered by the
  microwave manual and that you have no such information.
- Do not use your own knowledge and do not make up facts.
- Be concise and accurate; keep steps and numbers exactly as in the context.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION:
{query}"""


def main():
    embeddings_client = EmbeddingsClient(
        endpoint=OPENAI_EMBEDDINGS_ENDPOINT,
        model_name=OPENAI_EMBEDDINGS_MODEL,
        api_key=OPENAI_API_KEY
    )
    chat_client = ChatCompletionClient(
        endpoint=OPENAI_CHAT_COMPLETIONS_ENDPOINT,
        model_name=OPENAI_TERRA_MODEL,
        api_key=OPENAI_API_KEY
    )
    text_processor = TextProcessor(embeddings_client=embeddings_client, db_config=_DB_CONFIG)

    if input("Load context into DB? (y/n): ").strip().lower() == 'y':
        text_processor.process_text_file(
            file_name=_MANUAL_PATH, chunk_size=300, overlap=40, dimensions=_DIMENSIONS, truncate_table=True
        )

    conversation = Conversation()
    conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))

    print("🎯 Microwave RAG Assistant. Ask a question (type 'exit' to quit).")
    while True:
        user_question = input("\n> ").strip()
        if not user_question:
            continue
        if user_question.lower() in ("exit", "quit"):
            print("Bye!")
            break

        # Retrieval
        print("\n🔍 Retrieving context...")
        chunks = text_processor.search(
            search_mode=SearchMode.COSINE_DISTANCE,
            user_request=user_question,
            top_k=5,
            min_score=0.5,
            dimensions=_DIMENSIONS
        )
        context = "\n\n".join(chunks) if chunks else "No relevant context found."

        # Augmentation
        augmented_prompt = USER_PROMPT.format(context=context, query=user_question)
        print(f"\n🔗 Augmented prompt:\n{augmented_prompt}\n")
        conversation.add_message(Message(Role.USER, augmented_prompt))

        # Generation
        answer = chat_client.get_completion(conversation.get_messages())
        conversation.add_message(answer)
        print(f"🤖 Answer:\n{answer.content}")


if __name__ == "__main__":
    main()
