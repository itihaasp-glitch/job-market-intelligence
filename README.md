# Job Market Intelligence Pipeline

A Python-based pipeline that collects job postings from LinkedIn and Indeed,
normalises noisy real-world data, extracts skills using NLP, classifies roles,
and scores postings against a candidate resume profile.

## Features

- **Multi-source collection**: LinkedIn API + Indeed via Apify
- **Robust normalisation**: handles 9 date formats, salary extraction from prose,
  HTML stripping, deduplication
- **NLP skill extraction**: 200+ tech skills across 8 categories, longest-match algorithm
- **Role classification**: keyword-centroid scoring across 6 archetypes
- **Resume scoring**: weighted match (40% skills · 25% role · 20% seniority · 15% compat)
- **HTML report**: ranked job matches with apply links

## Tech Stack

Python · PostgreSQL · SQLite · AWS S3 · Parquet · Regex NLP · psycopg2 · boto3

## Project Structure
```
src/
├── job_collector.py   # API adapters, normalisation, deduplication
├── nlp_extractor.py   # Skill extraction, role classification, scoring
├── storage.py         # PostgreSQL + SQLite + S3 backends
└── pipeline.py        # Orchestrator, CLI, HTML report generator
```

## Quick Start
```bash
pip install requests psycopg2-binary boto3

export APIFY_TOKEN="your_token"

python src/pipeline.py \
  --queries "Data Engineer" "Senior Data Engineer" \
  --locations "Remote" "New York" \
  --resume resume.txt \
  --output-html report.html \
  --backend sqlite \
  --db-path jobs.db
```
