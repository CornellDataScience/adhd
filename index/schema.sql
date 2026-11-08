-- Database schema for RAG chunk storage
-- Matches contracts/chunk.schema.json. Safe to re-run.

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- Trigram matching for fuzzy search

-- Main chunks table
CREATE TABLE IF NOT EXISTS chunks (
    -- Identity (matches contract)
    id TEXT PRIMARY KEY,

    -- Source metadata
    repo TEXT NOT NULL,
    path TEXT NOT NULL,
    start_line INTEGER NOT NULL CHECK (start_line >= 1),
    end_line INTEGER NOT NULL CHECK (end_line >= 1),

    -- Code/doc metadata
    symbol TEXT,  -- NULL for chunks without a function/class
    language TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('code', 'doc', 'readme', 'file_tree', 'file_summary')),

    -- Version control
    commit_sha TEXT NOT NULL CHECK (commit_sha ~ '^[0-9a-f]{40}$'),
    source_url TEXT NOT NULL,
    file_hash TEXT NOT NULL CHECK (file_hash ~ '^[0-9a-f]{64}$'),

    -- Content
    token_count INTEGER NOT NULL CHECK (token_count >= 1),
    text TEXT NOT NULL,

    -- Search indexes (populated by index/ scripts)
    embedding vector(1024),  -- Qwen/Qwen3-Embedding-0.6B, 1024 dimensions
    embedding_model TEXT,    -- model that produced `embedding`, so vectors from different models never mix
    text_search tsvector,    -- Full-text search

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CHECK ((embedding IS NULL) = (embedding_model IS NULL))
);

-- Upgrade tables created by earlier versions of this file.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_model TEXT;
ALTER TABLE chunks ALTER COLUMN created_at TYPE TIMESTAMPTZ;
ALTER TABLE chunks ALTER COLUMN updated_at TYPE TIMESTAMPTZ;

-- Indexes for fast retrieval
-- No ANN index at init. Exact cosine until the table is large, then HNSW.
-- IVFFlat with lists = 100 assumes on the order of 100,000 rows; do not add it here.

-- 1. Full-text search
CREATE INDEX IF NOT EXISTS chunks_text_search_idx ON chunks
USING GIN(text_search);

-- 2. Trigram matching (fuzzy search for exact identifiers)
CREATE INDEX IF NOT EXISTS chunks_text_trgm_idx ON chunks
USING GIN(text gin_trgm_ops);

-- 3. Metadata indexes for filtering
CREATE INDEX IF NOT EXISTS chunks_repo_idx ON chunks(repo);
CREATE INDEX IF NOT EXISTS chunks_path_idx ON chunks(path);
CREATE INDEX IF NOT EXISTS chunks_kind_idx ON chunks(kind);
CREATE INDEX IF NOT EXISTS chunks_language_idx ON chunks(language);
CREATE INDEX IF NOT EXISTS chunks_file_hash_idx ON chunks(file_hash);

-- 4. Composite index for incremental updates
CREATE INDEX IF NOT EXISTS chunks_repo_path_idx ON chunks(repo, path);

-- Function to auto-update text_search column.
-- 'simple' keeps identifiers intact; 'english' stems get_user_settings to get/user/set.
CREATE OR REPLACE FUNCTION update_text_search()
RETURNS TRIGGER AS $$
BEGIN
    NEW.text_search = to_tsvector('simple', NEW.text);
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to keep text_search in sync
CREATE OR REPLACE TRIGGER chunks_text_search_update
BEFORE INSERT OR UPDATE ON chunks
FOR EACH ROW
EXECUTE FUNCTION update_text_search();

-- View for stats
CREATE OR REPLACE VIEW chunk_stats AS
SELECT
    repo,
    kind,
    language,
    COUNT(*) as chunk_count,
    AVG(token_count) as avg_tokens,
    MIN(token_count) as min_tokens,
    MAX(token_count) as max_tokens,
    SUM(token_count) as total_tokens
FROM chunks
GROUP BY repo, kind, language
ORDER BY repo, kind, language;

-- Grant permissions to the dev role, if it exists (see index/setup_local.sh)
DO $$
BEGIN
    IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'adhd') THEN
        GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO adhd;
        GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO adhd;
    END IF;
END
$$;
