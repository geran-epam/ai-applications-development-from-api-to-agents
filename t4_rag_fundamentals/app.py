import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

from commons.constants import OPENAI_API_KEY, OPENAI_EMBEDDINGS_MODEL, OPENAI_TERRA_MODEL, OPENAI_HOST

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MANUAL_PATH = os.path.join(_BASE_DIR, "microwave_manual.txt")
_INDEX_PATH = os.path.join(_BASE_DIR, "microwave_faiss_index")
# OpenRouter exposes an OpenAI-compatible API under /v1
_OPENROUTER_BASE_URL = f"{OPENAI_HOST}/v1"

#TODO:
# Create system prompt with:
# - role: explains the role for LLM and what it should do
# - Structure of User message, consists of 2 blocks:
#   - `RAG CONTEXT`: information retrieved on the Retrieval step based on user request
#   - `USER QUESTION`: The user's actual question
# - Instructions:
#   - Model must use only information from conversation
#   - Strictly forbid to answer questions that are not in the conversation or not present in `RAG CONTEXT`
_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a microwave oven using its user manual.

Each user message consists of 2 blocks:
- `##RAG CONTEXT:` information retrieved from the microwave manual based on the user's request
- `##USER QUESTION:` the user's actual question

Instructions:
- Use ONLY the information from the conversation, in particular from `RAG CONTEXT`, to answer.
- Strictly do NOT answer questions that are not related to the microwave or whose answer is not present in `RAG CONTEXT`.
  In that case reply that you can only answer questions covered by the microwave manual and that you have no such information.
- Do not use your own knowledge and do not make up facts.
- Be concise and accurate; keep steps and numbers exactly as in the context.
"""

_USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION:
{query}"""


class MicrowaveRAG:

    def __init__(self, embeddings: OpenAIEmbeddings, llm_client: ChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        """
        Load existing FAISS index from disk or create a new one.
        Returns:
              VectorStore: Initialized FAISS vectorstore.
        """
        #TODO:
        # - Print a startup message
        # - Check if 'microwave_faiss_index' folder already exists
        # - If yes, load the index from disk using FAISS.load_local()
        # - If no, call _create_new_index() to build and save a fresh index
        # - Return the vectorstore
        print("🔄 Initializing vector store...")
        if os.path.exists(_INDEX_PATH):
            print("📂 Loading existing FAISS index...")
            vectorstore = FAISS.load_local(
                _INDEX_PATH, self.embeddings, allow_dangerous_deserialization=True
            )
        else:
            print("🆕 No index found, creating a new one...")
            vectorstore = self._create_new_index()
        return vectorstore

    def _create_new_index(self) -> VectorStore:
        """
        Load the manual, split into chunks, embed, and save a new FAISS index.
        Returns:
              VectorStore: Newly created and saved FAISS vectorstore.
        """
        #TODO:
        # - Load 'microwave_manual.txt' using TextLoader
        # - Split documents into chunks using RecursiveCharacterTextSplitter
        #   (chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", "."])
        # - Create a FAISS vectorstore from chunks and self.embeddings using FAISS.from_documents()
        # - Save the index locally using vectorstore.save_local("microwave_faiss_index")
        # - Return the vectorstore
        documents = TextLoader(_MANUAL_PATH, encoding="utf-8").load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", "."]
        )
        chunks = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        vectorstore.save_local(_INDEX_PATH)
        print(f"✅ Index created with {len(chunks)} chunks and saved to '{_INDEX_PATH}'")
        return vectorstore

    def retrieve_context(self, query: str, k: int = 4, score=0.3):
        """
        Retrieve the context for a given query.
        Args:
              query (str): The query to retrieve the context for.
              k (int): The number of relevant documents(chunks) to retrieve.
              score (float): The similarity score between documents and query. Range 0.0 to 1.0.
        """
        #TODO:
        # - Search the vectorstore using similarity_search_with_relevance_scores() with k and score_threshold parameters
        # - Iterate over results, collect each doc's page_content, and print its relevance score
        # - Return all collected chunks joined with "\n\n" as a single context string
        print(f"\n🔍 Retrieving context (k={k}, score_threshold={score})...")
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=k, score_threshold=score
        )
        chunks = []
        for doc, relevance in results:
            chunks.append(doc.page_content)
            print(f"  - relevance score: {relevance:.4f}")
            print(f"    {doc.page_content[:100]!r}...")
        if not chunks:
            print("  (no relevant chunks found)")
        return "\n\n".join(chunks)

    def augment_prompt(self, query: str, context: str):
        """
        Inject retrieved context and user query into the prompt template.
        Args:
              query (str): The user's question.
              context (str): Retrieved context from the vectorstore.
        Returns:
              str: Formatted prompt ready for the LLM.
        """
        #TODO:
        # - Format _USER_PROMPT template substituting {context} and {query}
        # - Print the resulting augmented prompt
        # - Return the formatted string
        augmented_prompt = _USER_PROMPT.format(context=context, query=query)
        print(f"\n🔗 Augmented prompt:\n{augmented_prompt}\n")
        return augmented_prompt

    def generate_answer(self, augmented_prompt: str):
        """
        Send the augmented prompt to the LLM and return its response.
        Args:
              augmented_prompt (str): The prompt with injected context and query.
        Returns:
              str: The LLM-generated answer.
        """
        #TODO:
        # - Build a messages list: [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=augmented_prompt)]
        # - Invoke self.llm_client with the messages list
        # - Print the response content
        # - Return the response content string
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt),
        ]
        response = self.llm_client.invoke(messages)
        print(f"🤖 Answer:\n{response.content}\n")
        return response.content


def main(rag: MicrowaveRAG):
    #TODO:
    # - Print a welcome message
    # - Run an infinite loop that reads user input with input()
    # - For each question execute the 3-step RAG pipeline:
    #   - Step 1 (Retrieval):   call rag.retrieve_context() to fetch relevant chunks
    #   - Step 2 (Augmentation): call rag.augment_prompt() to build the prompt
    #   - Step 3 (Generation):  call rag.generate_answer() to get the LLM answer
    print("🎯 Microwave Manual RAG Assistant. Ask a question about the microwave (type 'exit' to quit).")
    while True:
        user_question = input("\n> ").strip()
        if not user_question:
            continue
        if user_question.lower() in ("exit", "quit"):
            print("Bye!")
            break

        # Step 1: Retrieval
        context = rag.retrieve_context(user_question)
        # Step 2: Augmentation
        augmented_prompt = rag.augment_prompt(user_question, context)
        # Step 3: Generation
        rag.generate_answer(augmented_prompt)


if __name__ == "__main__":
    main(
        MicrowaveRAG(
            embeddings=OpenAIEmbeddings(
                model=OPENAI_EMBEDDINGS_MODEL,
                api_key=SecretStr(OPENAI_API_KEY),
                base_url=_OPENROUTER_BASE_URL,
                # OpenRouter expects raw text, not pre-tokenized input
                check_embedding_ctx_length=False,
            ),
            llm_client=ChatOpenAI(
                temperature=0.0,
                reasoning_effort="none",
                model=OPENAI_TERRA_MODEL,
                api_key=SecretStr(OPENAI_API_KEY),
                base_url=_OPENROUTER_BASE_URL,
            ),
        )
    )