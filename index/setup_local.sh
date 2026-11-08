#!/usr/bin/env bash
# Create the local role and database that retrieval/retrieve.py expects, then apply schema.sql.
# Safe to re-run. Usage: index/setup_local.sh
set -euo pipefail

PG_BIN=${PG_BIN:-$(dirname "$(command -v psql || echo /opt/homebrew/opt/postgresql@17/bin/psql)")}
PSQL=("$PG_BIN/psql" -h localhost -v ON_ERROR_STOP=1 -q)
DB=${DB:-adhd_rag}
export PGOPTIONS="-c client_min_messages=warning"  # hide "already exists, skipping" notices

"${PSQL[@]}" -d postgres <<'SQL'
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'adhd') THEN
        CREATE ROLE adhd LOGIN PASSWORD 'dev_password_change_in_prod';
    END IF;
END
$$;
SQL

if ! "${PSQL[@]}" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = '$DB'" | grep -q 1; then
    "$PG_BIN/createdb" -h localhost -O adhd "$DB"
fi

"${PSQL[@]}" -d "$DB" -f "$(dirname "$0")/schema.sql"
echo "Ready: postgresql://adhd:dev_password_change_in_prod@localhost:5432/$DB"
