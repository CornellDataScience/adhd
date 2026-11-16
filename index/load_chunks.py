"""Load chunks from JSONL into Postgres database.

Usage:
    uv run python index/load_chunks.py data/chunks/all_repos.jsonl
    uv run python index/load_chunks.py data/chunks/adhd.jsonl --incremental
"""

import argparse
import json
import sys
from pathlib import Path

import psycopg


def get_db_connection(db_url: str = "postgresql://adhd:dev_password_change_in_prod@localhost:5432/adhd_rag"):
    """Connect to Postgres database."""
    try:
        return psycopg.connect(db_url)
    except psycopg.OperationalError as e:
        print(f"❌ Failed to connect to database: {e}")
        print("\nMake sure Postgres is running:")
        print("  docker compose up -d")
        sys.exit(1)


def load_chunks(jsonl_path: Path, incremental: bool = False, db_url: str = None):
    """Load chunks from JSONL into database."""

    if not jsonl_path.exists():
        print(f"❌ File not found: {jsonl_path}")
        sys.exit(1)

    # Read chunks
    chunks = []
    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunk = json.loads(line)
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping invalid JSON on line {line_num}: {e}")

    if not chunks:
        print("❌ No valid chunks found")
        sys.exit(1)

    print(f"📦 Loaded {len(chunks)} chunks from {jsonl_path}")

    # Connect to database
    conn = get_db_connection(db_url)

    with conn.cursor() as cur:
        if not incremental:
            # Full reload: delete all existing chunks
            print("🗑️  Clearing existing chunks...")
            cur.execute("DELETE FROM chunks")
            conn.commit()

        # Insert chunks
        inserted = 0
        updated = 0
        skipped = 0

        for chunk in chunks:
            try:
                # Check required fields
                required = ['id', 'repo', 'path', 'start_line', 'end_line',
                           'language', 'kind', 'commit_sha', 'source_url',
                           'file_hash', 'token_count', 'text']

                for field in required:
                    if field not in chunk:
                        print(f"⚠️  Chunk {chunk.get('id', 'unknown')} missing field: {field}")
                        skipped += 1
                        continue

                # Insert or update
                cur.execute("""
                    INSERT INTO chunks (
                        id, repo, path, start_line, end_line, symbol,
                        language, kind, commit_sha, source_url, file_hash,
                        token_count, text
                    ) VALUES (
                        %(id)s, %(repo)s, %(path)s, %(start_line)s, %(end_line)s, %(symbol)s,
                        %(language)s, %(kind)s, %(commit_sha)s, %(source_url)s, %(file_hash)s,
                        %(token_count)s, %(text)s
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        repo = EXCLUDED.repo,
                        path = EXCLUDED.path,
                        start_line = EXCLUDED.start_line,
                        end_line = EXCLUDED.end_line,
                        symbol = EXCLUDED.symbol,
                        language = EXCLUDED.language,
                        kind = EXCLUDED.kind,
                        commit_sha = EXCLUDED.commit_sha,
                        source_url = EXCLUDED.source_url,
                        file_hash = EXCLUDED.file_hash,
                        token_count = EXCLUDED.token_count,
                        text = EXCLUDED.text,
                        updated_at = NOW()
                """, chunk)

                if cur.rowcount == 1:
                    inserted += 1
                else:
                    updated += 1

            except Exception as e:
                print(f"⚠️  Failed to insert chunk {chunk.get('id', 'unknown')}: {e}")
                skipped += 1

        conn.commit()

    conn.close()

    # Print summary
    print("\n✅ Loading complete!")
    print(f"   Inserted: {inserted}")
    print(f"   Updated:  {updated}")
    print(f"   Skipped:  {skipped}")
    print(f"   Total:    {len(chunks)}")

    # Print stats
    conn = get_db_connection(db_url)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks")
        total = cur.fetchone()[0]

        cur.execute("""
            SELECT repo, COUNT(*), AVG(token_count)::int
            FROM chunks
            GROUP BY repo
            ORDER BY COUNT(*) DESC
        """)

        print("\n📊 Database stats:")
        print(f"   Total chunks: {total}")
        print("\n   By repository:")
        for row in cur.fetchall():
            print(f"     {row[0]}: {row[1]} chunks (avg {row[2]} tokens)")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Load chunks from JSONL into Postgres")
    parser.add_argument("jsonl_path", type=Path, help="Path to chunks JSONL file")
    parser.add_argument("--incremental", action="store_true",
                       help="Incremental load (don't delete existing chunks)")
    parser.add_argument("--db-url", default=None,
                       help="Database URL (default: postgresql://adhd:dev_password_change_in_prod@localhost:5432/adhd_rag)")

    args = parser.parse_args()

    load_chunks(args.jsonl_path, args.incremental, args.db_url)


if __name__ == "__main__":
    main()
