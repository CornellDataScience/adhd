"""Test database connection and basic operations.

Usage:
    uv run python index/test_db.py
"""

import psycopg


DB_URL = "postgresql://adhd:dev_password_change_in_prod@localhost:5432/adhd_rag"


def test_connection():
    """Test database connection."""
    print("🔌 Testing database connection...")

    try:
        conn = psycopg.connect(DB_URL)
        print("✅ Connected to database!")

        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
            print(f"   PostgreSQL version: {version[:50]}...")

        conn.close()
        return True

    except psycopg.OperationalError as e:
        print(f"❌ Connection failed: {e}")
        print("\nMake sure Postgres is running:")
        print("  docker compose up -d")
        return False


def test_extensions():
    """Test pgvector and pg_trgm extensions."""
    print("\n🧩 Testing extensions...")

    conn = psycopg.connect(DB_URL)

    with conn.cursor() as cur:
        # Check pgvector
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        if cur.fetchone():
            print("✅ pgvector extension installed")
        else:
            print("❌ pgvector extension NOT found")

        # Check pg_trgm
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'pg_trgm'")
        if cur.fetchone():
            print("✅ pg_trgm extension installed")
        else:
            print("❌ pg_trgm extension NOT found")

    conn.close()


def test_schema():
    """Test that chunks table exists with correct schema."""
    print("\n📋 Testing schema...")

    conn = psycopg.connect(DB_URL)

    with conn.cursor() as cur:
        # Check table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'chunks'
            )
        """)

        if cur.fetchone()[0]:
            print("✅ chunks table exists")

            # Check columns
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'chunks'
                ORDER BY ordinal_position
            """)

            print("\n   Columns:")
            for row in cur.fetchall():
                print(f"     {row[0]:<20} {row[1]}")

        else:
            print("❌ chunks table NOT found")
            print("\nRun migrations:")
            print("  docker compose down -v  # Reset database")
            print("  docker compose up -d    # Restart with schema.sql")

    conn.close()


def test_insert_and_search():
    """Test inserting a sample chunk and searching."""
    print("\n🔬 Testing insert and search...")

    conn = psycopg.connect(DB_URL)

    # Sample chunk
    test_chunk = {
        "id": "test_chunk_123",
        "repo": "CornellDataScience/adhd",
        "path": "test.py",
        "start_line": 1,
        "end_line": 5,
        "symbol": "test_function",
        "language": "python",
        "kind": "code",
        "commit_sha": "0" * 40,
        "source_url": "https://github.com/CornellDataScience/adhd/blob/main/test.py",
        "file_hash": "0" * 64,
        "token_count": 50,
        "text": "# Test chunk\ndef test_function():\n    return 'hello world'"
    }

    with conn.cursor() as cur:
        # Insert
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
            ON CONFLICT (id) DO NOTHING
        """, test_chunk)

        conn.commit()
        print("✅ Inserted test chunk")

        # Full-text search
        cur.execute("""
            SELECT id, repo, path, symbol
            FROM chunks
            WHERE text_search @@ to_tsquery('english', 'test')
            LIMIT 5
        """)

        results = cur.fetchall()
        if results:
            print(f"✅ Full-text search works ({len(results)} results)")
            for row in results[:3]:
                print(f"     {row[1]}/{row[2]} - {row[3]}")
        else:
            print("⚠️  No results from full-text search")

        # Clean up test chunk
        cur.execute("DELETE FROM chunks WHERE id = 'test_chunk_123'")
        conn.commit()

    conn.close()


def show_stats():
    """Show database statistics."""
    print("\n📊 Database statistics...")

    conn = psycopg.connect(DB_URL)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks")
        total = cur.fetchone()[0]
        print(f"   Total chunks: {total}")

        if total > 0:
            cur.execute("""
                SELECT
                    COUNT(DISTINCT repo) as repos,
                    COUNT(DISTINCT language) as languages,
                    AVG(token_count)::int as avg_tokens,
                    MIN(token_count) as min_tokens,
                    MAX(token_count) as max_tokens
                FROM chunks
            """)

            row = cur.fetchone()
            print(f"   Repositories: {row[0]}")
            print(f"   Languages: {row[1]}")
            print(f"   Token count: {row[3]} - {row[4]} (avg: {row[2]})")

    conn.close()


def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 ADHD RAG Database Tests")
    print("=" * 60)

    if not test_connection():
        return

    test_extensions()
    test_schema()
    test_insert_and_search()
    show_stats()

    print("\n" + "=" * 60)
    print("✨ All tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
