"""
storage.py
----------
Storage layer for the Job Intelligence Pipeline.

Backends
========
- PostgreSQL (via psycopg2) — primary relational store for search/analytics
- S3 (via boto3)            — raw payload archive + Parquet exports
- In-memory (SQLite)        — local dev / unit tests (no external deps)

Schema design for search
========================
The main table uses a GIN index on skills (text[]) and a tsvector column on
title + description for full-text search, enabling fast queries like:
  "Data Engineer roles requiring Kafka in US with salary > $120k"
"""

import json
import logging
import os
import sqlite3
import gzip
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

POSTGRES_DDL = """
-- Job postings (primary store)
CREATE TABLE IF NOT EXISTS job_postings (
    posting_id          TEXT PRIMARY KEY,
    external_id         TEXT NOT NULL,
    source              TEXT NOT NULL,
    title               TEXT NOT NULL,
    company             TEXT,
    location            TEXT,
    remote_type         TEXT,
    employment_type     TEXT,
    salary_min          NUMERIC,
    salary_max          NUMERIC,
    salary_currency     CHAR(3),
    salary_period       TEXT,
    description_clean   TEXT,
    skills              TEXT[],          -- array for GIN index
    role_category       TEXT,
    seniority           TEXT,
    visa_sponsored      BOOLEAN,
    posted_at           TIMESTAMPTZ,
    collected_at        TIMESTAMPTZ DEFAULT NOW(),
    apply_url           TEXT,

    -- full-text search vector (maintained by trigger)
    fts_vector          TSVECTOR,

    CONSTRAINT uq_source_external UNIQUE (source, external_id)
);

-- GIN index: fast skill overlap queries  →  WHERE skills && ARRAY['kafka','spark']
CREATE INDEX IF NOT EXISTS idx_postings_skills
    ON job_postings USING GIN (skills);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_postings_fts
    ON job_postings USING GIN (fts_vector);

-- B-tree indexes for common filter columns
CREATE INDEX IF NOT EXISTS idx_postings_role     ON job_postings (role_category);
CREATE INDEX IF NOT EXISTS idx_postings_seniority ON job_postings (seniority);
CREATE INDEX IF NOT EXISTS idx_postings_remote   ON job_postings (remote_type);
CREATE INDEX IF NOT EXISTS idx_postings_collected ON job_postings (collected_at DESC);

-- Trigger to maintain fts_vector automatically
CREATE OR REPLACE FUNCTION update_fts()
RETURNS TRIGGER AS $$
BEGIN
    NEW.fts_vector :=
        setweight(to_tsvector('english', COALESCE(NEW.title, '')),       'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.company, '')),     'B') ||
        setweight(to_tsvector('english', COALESCE(array_to_string(NEW.skills, ' '), '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(NEW.description_clean, '')), 'C');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trig_update_fts ON job_postings;
CREATE TRIGGER trig_update_fts
BEFORE INSERT OR UPDATE ON job_postings
FOR EACH ROW EXECUTE FUNCTION update_fts();


-- Match results (for resume-based scoring)
CREATE TABLE IF NOT EXISTS match_results (
    id                  BIGSERIAL PRIMARY KEY,
    posting_id          TEXT REFERENCES job_postings(posting_id) ON DELETE CASCADE,
    resume_hash         TEXT NOT NULL,      -- SHA256 of resume text
    score               NUMERIC(5,1) NOT NULL,
    matched_skills      TEXT[],
    missing_skills      TEXT[],
    title_relevance     NUMERIC(4,2),
    skill_overlap       NUMERIC(4,2),
    seniority_fit       NUMERIC(4,2),
    visa_ok             BOOLEAN,
    remote_ok           BOOLEAN,
    breakdown           JSONB,
    scored_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_match_score
    ON match_results (resume_hash, score DESC);
"""

SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS job_postings (
    posting_id          TEXT PRIMARY KEY,
    external_id         TEXT NOT NULL,
    source              TEXT NOT NULL,
    title               TEXT NOT NULL,
    company             TEXT,
    location            TEXT,
    remote_type         TEXT,
    employment_type     TEXT,
    salary_min          REAL,
    salary_max          REAL,
    salary_currency     TEXT,
    salary_period       TEXT,
    description_clean   TEXT,
    skills              TEXT,            -- JSON array string in SQLite
    role_category       TEXT,
    seniority           TEXT,
    visa_sponsored      INTEGER,
    posted_at           TEXT,
    collected_at        TEXT,
    apply_url           TEXT
);

CREATE TABLE IF NOT EXISTS match_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    posting_id          TEXT,
    resume_hash         TEXT,
    score               REAL,
    matched_skills      TEXT,            -- JSON
    missing_skills      TEXT,
    title_relevance     REAL,
    skill_overlap       REAL,
    seniority_fit       REAL,
    visa_ok             INTEGER,
    remote_ok           INTEGER,
    breakdown           TEXT,
    scored_at           TEXT
);
"""


# ---------------------------------------------------------------------------
# Postgres backend
# ---------------------------------------------------------------------------

class PostgresStore:
    def __init__(self, dsn: str):
        import psycopg2
        import psycopg2.extras
        self._psycopg2 = psycopg2
        self._extras   = psycopg2.extras
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = False
        self._init_schema()

    def _init_schema(self):
        with self.conn.cursor() as cur:
            cur.execute(POSTGRES_DDL)
        self.conn.commit()
        logger.info("PostgreSQL schema initialised")

    def upsert_postings(self, postings: list) -> int:
        sql = """
            INSERT INTO job_postings
                (posting_id, external_id, source, title, company, location,
                 remote_type, employment_type, salary_min, salary_max,
                 salary_currency, salary_period, description_clean, skills,
                 role_category, seniority, visa_sponsored, posted_at,
                 collected_at, apply_url)
            VALUES %s
            ON CONFLICT (source, external_id) DO UPDATE SET
                title             = EXCLUDED.title,
                skills            = EXCLUDED.skills,
                role_category     = EXCLUDED.role_category,
                seniority         = EXCLUDED.seniority,
                salary_min        = EXCLUDED.salary_min,
                salary_max        = EXCLUDED.salary_max,
                description_clean = EXCLUDED.description_clean,
                collected_at      = EXCLUDED.collected_at
        """
        rows = [
            (
                p.posting_id, p.external_id, p.source, p.title, p.company,
                p.location, p.remote_type, p.employment_type,
                p.salary_min, p.salary_max, p.salary_currency, p.salary_period,
                p.description_clean[:10_000],   # cap at 10k chars
                p.skills, p.role_category, p.seniority, p.visa_sponsored,
                p.posted_at, p.collected_at, p.apply_url,
            )
            for p in postings
        ]
        with self.conn.cursor() as cur:
            self._extras.execute_values(cur, sql, rows, page_size=500)
        self.conn.commit()
        logger.info("Upserted %d postings", len(rows))
        return len(rows)

    def search(
        self,
        keywords: Optional[str] = None,
        skills: Optional[List[str]] = None,
        role: Optional[str] = None,
        seniority: Optional[str] = None,
        remote_type: Optional[str] = None,
        visa_sponsored: Optional[bool] = None,
        salary_min: Optional[float] = None,
        limit: int = 50,
    ) -> List[Dict]:
        clauses, params = [], []

        if keywords:
            clauses.append("fts_vector @@ plainto_tsquery('english', %s)")
            params.append(keywords)
        if skills:
            clauses.append("skills && %s")
            params.append(skills)
        if role:
            clauses.append("role_category = %s")
            params.append(role)
        if seniority:
            clauses.append("seniority = %s")
            params.append(seniority)
        if remote_type:
            clauses.append("remote_type = %s")
            params.append(remote_type)
        if visa_sponsored is not None:
            clauses.append("visa_sponsored = %s")
            params.append(visa_sponsored)
        if salary_min is not None:
            clauses.append("(salary_min >= %s OR salary_max >= %s)")
            params.extend([salary_min, salary_min])

        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        sql = f"""
            SELECT posting_id, title, company, location, remote_type,
                   seniority, role_category, skills, salary_min, salary_max,
                   salary_currency, visa_sponsored, apply_url, posted_at
            FROM job_postings
            {where}
            ORDER BY collected_at DESC
            LIMIT %s
        """
        params.append(limit)

        with self.conn.cursor(cursor_factory=self._extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def save_match_results(self, results: list, resume_hash: str) -> None:
        sql = """
            INSERT INTO match_results
                (posting_id, resume_hash, score, matched_skills, missing_skills,
                 title_relevance, skill_overlap, seniority_fit, visa_ok, remote_ok, breakdown)
            VALUES %s
            ON CONFLICT DO NOTHING
        """
        rows = [
            (r.posting_id, resume_hash, r.score, r.matched_skills, r.missing_skills,
             r.title_relevance, r.skill_overlap, r.seniority_fit,
             r.visa_ok, r.remote_ok, json.dumps(r.breakdown))
            for r in results
        ]
        with self.conn.cursor() as cur:
            self._extras.execute_values(cur, sql, rows)
        self.conn.commit()


# ---------------------------------------------------------------------------
# SQLite backend (local / test)
# ---------------------------------------------------------------------------

class SQLiteStore:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(SQLITE_DDL)
        self.conn.commit()

    def upsert_postings(self, postings: list) -> int:
        sql = """
            INSERT OR REPLACE INTO job_postings
                (posting_id, external_id, source, title, company, location,
                 remote_type, employment_type, salary_min, salary_max,
                 salary_currency, salary_period, description_clean, skills,
                 role_category, seniority, visa_sponsored, posted_at,
                 collected_at, apply_url)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        rows = [
            (
                p.posting_id, p.external_id, p.source, p.title, p.company,
                p.location, p.remote_type, p.employment_type,
                p.salary_min, p.salary_max, p.salary_currency, p.salary_period,
                p.description_clean[:10_000],
                json.dumps(p.skills), p.role_category, p.seniority,
                int(p.visa_sponsored),
                p.posted_at.isoformat() if p.posted_at else None,
                p.collected_at.isoformat(),
                p.apply_url,
            )
            for p in postings
        ]
        self.conn.executemany(sql, rows)
        self.conn.commit()
        return len(rows)

    def search(self, role: Optional[str] = None, skills: Optional[List[str]] = None,
               limit: int = 50) -> List[Dict]:
        clauses, params = [], []
        if role:
            clauses.append("role_category = ?")
            params.append(role)
        if skills:
            for sk in skills:
                clauses.append("skills LIKE ?")
                params.append(f'%"{sk}"%')
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        sql = f"SELECT * FROM job_postings {where} ORDER BY collected_at DESC LIMIT ?"
        params.append(limit)
        cur = self.conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# S3 archive
# ---------------------------------------------------------------------------

class S3Archive:
    """
    Archives raw payloads as gzip-compressed NDJSON, partitioned by date/source.
    Also exports Parquet snapshots for Athena/Redshift Spectrum queries.
    """

    def __init__(self, bucket: str, prefix: str = "job_postings", region: str = "us-east-1"):
        import boto3
        self.s3     = boto3.client("s3", region_name=region)
        self.bucket = bucket
        self.prefix = prefix

    def archive_raw(self, postings: list) -> str:
        """Write gzip NDJSON to s3://{bucket}/{prefix}/raw/{date}/{source}/batch.ndjson.gz"""
        if not postings:
            return ""
        source = postings[0].source
        date   = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        ts     = int(datetime.now(timezone.utc).timestamp())
        key    = f"{self.prefix}/raw/{date}/{source}/{ts}.ndjson.gz"

        buf = BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            for p in postings:
                line = json.dumps(p.raw_payload) + "\n"
                gz.write(line.encode("utf-8"))

        buf.seek(0)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buf.read(),
            ContentEncoding="gzip",
            ContentType="application/x-ndjson",
        )
        logger.info("Archived %d raw payloads → s3://%s/%s", len(postings), self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def export_parquet(self, postings: list) -> str:
        """Export enriched postings as Parquet via pandas + pyarrow."""
        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            logger.warning("pandas/pyarrow not installed; skipping Parquet export")
            return ""

        records = [p.to_dict() for p in postings]
        df = pd.DataFrame(records)
        buf = BytesIO()
        table = pa.Table.from_pandas(df)
        pq.write_table(table, buf, compression="snappy")
        buf.seek(0)

        date = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        key  = f"{self.prefix}/parquet/dt={date}/postings.parquet"
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=buf.read())
        logger.info("Exported Parquet → s3://%s/%s", self.bucket, key)
        return f"s3://{self.bucket}/{key}"


# ---------------------------------------------------------------------------
# Store factory
# ---------------------------------------------------------------------------

def get_store(backend: str = "sqlite", **kwargs):
    if backend == "postgres":
        dsn = kwargs.get("dsn") or os.environ["POSTGRES_DSN"]
        return PostgresStore(dsn)
    return SQLiteStore(kwargs.get("db_path", ":memory:"))
