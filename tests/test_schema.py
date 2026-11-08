import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

SCHEMA = Path(__file__).parent.parent / "index" / "schema.sql"
PSQL = shutil.which("psql") or "/opt/homebrew/opt/postgresql@17/bin/psql"


def psql(db, *args, check=True):
    return subprocess.run([PSQL, "-h", "localhost", "-d", db, "-v", "ON_ERROR_STOP=1", "-Atq", *args], capture_output=True, text=True, check=check)


def postgres_available():
    try:
        return psql("postgres", "-c", "SELECT 1", check=False).returncode == 0
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not postgres_available(), reason="no local Postgres")


@pytest.fixture
def db():
    name = f"schema_test_{uuid.uuid4().hex[:8]}"
    psql("postgres", "-c", f"CREATE DATABASE {name}")
    yield name
    psql("postgres", "-c", f"DROP DATABASE {name}")


ROW = """INSERT INTO chunks (id, repo, path, start_line, end_line, symbol, language, kind, commit_sha, source_url, file_hash, token_count, text)
VALUES ('t1', 'org/r', 'a.py', 1, 2, NULL, 'python', 'code', repeat('a', 40), 'https://x', repeat('b', 64), 5, 'def get_user_settings(): pass')"""


def test_schema_applies_twice(db):
    psql(db, "-f", str(SCHEMA))
    psql(db, "-f", str(SCHEMA))


def test_text_search_keeps_identifiers(db):
    psql(db, "-f", str(SCHEMA))
    psql(db, "-c", ROW)
    assert "'settings'" in psql(db, "-c", "SELECT text_search FROM chunks").stdout


def test_embedding_requires_model(db):
    psql(db, "-f", str(SCHEMA))
    psql(db, "-c", ROW)
    vector = "array_fill(0.1, ARRAY[1024])::vector"
    assert psql(db, "-c", f"UPDATE chunks SET embedding = {vector}", check=False).returncode != 0
    psql(db, "-c", f"UPDATE chunks SET embedding = {vector}, embedding_model = 'm'")
