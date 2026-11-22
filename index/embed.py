"""Fill chunks.embedding for rows that do not have a vector yet.

The schema requires embedding and embedding_model to be set together.
Documents are embedded as stored chunk text (no query instruction).

Usage:
    uv run python index/embed.py
    uv run python index/embed.py --force --batch-size 8
"""

import argparse
import sys

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from retrieval.embed import DIMENSIONS, MODEL_ID, embed_documents
from retrieval.retrieve import resolve_database_url


def write_embeddings(database_url: str, *, force: bool = False, batch_size: int = 8) -> int:
    """Write 1024-d vectors. Returns how many rows were updated.

    By default only rows with a null embedding are embedded. ``force``
    re-embeds every row and replaces ``embedding_model``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    conn = psycopg.connect(database_url)
    try:
        register_vector(conn)
        sql = "SELECT id, text FROM chunks"
        if not force:
            sql += " WHERE embedding IS NULL"
        sql += " ORDER BY id"
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        if not rows:
            return 0

        written = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            vectors = embed_documents([text for _chunk_id, text in batch])
            with conn.cursor() as cur:
                for (chunk_id, _text), vector in zip(batch, vectors, strict=True):
                    if len(vector) != DIMENSIONS:
                        raise ValueError(f"{MODEL_ID} returned {len(vector)} dimensions, expected {DIMENSIONS}")
                    cur.execute(
                        """
                        UPDATE chunks
                        SET embedding = %s, embedding_model = %s
                        WHERE id = %s
                        """,
                        (Vector(vector), MODEL_ID, chunk_id),
                    )
            conn.commit()
            written += len(batch)
        return written
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Embed chunks with Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--force", action="store_true", help="Re-embed every row, not only null embeddings")
    parser.add_argument("--batch-size", type=int, default=8, help="Chunks per forward pass (default: 8)")
    parser.add_argument("--db-url", default=None, help="Postgres URL (default: DATABASE_URL or the local adhd_rag URL)")
    args = parser.parse_args()

    database_url = resolve_database_url(args.db_url)
    try:
        written = write_embeddings(database_url, force=args.force, batch_size=args.batch_size)
    except psycopg.OperationalError as exc:
        print(f"Failed to connect to database: {exc}", file=sys.stderr)
        print("Apply index/schema.sql first (index/setup_local.sh, or docker compose up -d).", file=sys.stderr)
        sys.exit(1)
    print(f"Embedded {written} chunks with {MODEL_ID}")


if __name__ == "__main__":
    main()
