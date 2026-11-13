import os
from enum import StrEnum

import psycopg2
from psycopg2.extras import RealDictCursor

from t5_rag_advanced.embeddings.embeddings_client import EmbeddingsClient
from t5_rag_advanced.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: EmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def process_text_file(
            self,
            file_name: str,
            chunk_size: int = 300,
            overlap: int = 40,
            dimensions: int = 384,
            truncate_table: bool = True
    ):
        """Load file, split into chunks, generate embeddings and store them in DB"""
        if truncate_table:
            self._truncate_table()

        with open(file_name, 'r', encoding='utf-8') as file:
            content = file.read()

        chunks = chunk_text(content, chunk_size, overlap)
        if not chunks:
            print("⚠️ No chunks to process")
            return
        embeddings = self.embeddings_client.get_embeddings(chunks, dimensions)

        document_name = os.path.basename(file_name)[:64]
        for index, chunk in enumerate(chunks):
            self._save_chunk(document_name, chunk, embeddings[index])
        print(f"✅ Saved {len(chunks)} chunks from '{document_name}' to DB")

    def _truncate_table(self):
        """Remove all rows from the vectors table"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE vectors RESTART IDENTITY")

    def _save_chunk(self, document_name: str, text: str, embedding: list[float]):
        """Insert a chunk with its embedding into DB"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector)",
                    (document_name, text, str(embedding))
                )

    def search(
            self,
            search_mode: SearchMode,
            user_request: str,
            top_k: int = 5,
            min_score: float = 0.5,
            dimensions: int = 384
    ) -> list[str]:
        """Find the most relevant chunks. Chunks with distance > min_score are filtered out"""
        embeddings = self.embeddings_client.get_embeddings(user_request, dimensions)
        query_vector = str(embeddings[0])

        operator = "<=>" if search_mode == SearchMode.COSINE_DISTANCE else "<->"
        query = f"""
            SELECT text, embedding {operator} %s::vector AS distance
            FROM vectors
            WHERE embedding {operator} %s::vector <= %s
            ORDER BY distance
            LIMIT %s
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (query_vector, query_vector, min_score, top_k))
                rows = cursor.fetchall()

        for row in rows:
            print(f"  - distance={row['distance']:.4f}: {row['text'][:80]!r}...")
        return [row['text'] for row in rows]
