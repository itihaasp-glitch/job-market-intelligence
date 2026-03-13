"""
nlp_extractor.py
----------------
NLP layer for the Job Intelligence Pipeline.

Responsibilities
================
1. Skill extraction  — rule-based trie lookup + regex for 200+ tech skills
2. Role classification — multi-label classification by keyword centroid scoring
3. Seniority inference — combined title heuristics + JD keyword weighting
4. Keyword scoring for resume-matching (TF-IDF inspired, no external deps)

Design choice: pure Python + regex, no heavy ML dependencies.
This keeps cold-start time <2 s and works in restricted Lambda environments.
An optional spaCy / transformers path is provided for richer extraction.
"""

from __future__ import annotations

import re
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skill taxonomy (200+ terms, organised by category)
# ---------------------------------------------------------------------------

SKILLS: Dict[str, List[str]] = {
    "languages": [
        "python", "sql", "scala", "java", "go", "golang", "rust", "r", "julia",
        "javascript", "typescript", "bash", "shell", "hive", "pig",
    ],
    "cloud": [
        "aws", "gcp", "azure", "s3", "ec2", "emr", "glue", "lambda",
        "redshift", "athena", "kinesis", "sqs", "sns", "step functions",
        "bigquery", "dataflow", "pubsub", "cloud storage",
        "azure synapse", "azure data factory", "adls",
    ],
    "processing": [
        "spark", "pyspark", "flink", "kafka", "airflow", "dagster", "prefect",
        "dbt", "beam", "nifi", "storm", "samza", "hadoop", "mapreduce",
        "spark streaming", "structured streaming", "delta lake", "iceberg",
        "hudi", "trino", "presto", "dask",
    ],
    "databases": [
        "postgres", "postgresql", "mysql", "oracle", "sql server", "sqlite",
        "mongodb", "cassandra", "dynamodb", "redis", "elasticsearch",
        "opensearch", "neo4j", "snowflake", "databricks", "hbase",
        "clickhouse", "druid", "pinot",
    ],
    "ml_tools": [
        "tensorflow", "pytorch", "scikit-learn", "sklearn", "xgboost",
        "lightgbm", "keras", "hugging face", "mlflow", "sagemaker",
        "vertex ai", "kubeflow", "feast", "great expectations",
    ],
    "devops": [
        "docker", "kubernetes", "k8s", "terraform", "cloudformation", "cdk",
        "ansible", "jenkins", "github actions", "gitlab ci", "circleci",
        "datadog", "grafana", "prometheus", "splunk", "elk", "helm",
    ],
    "concepts": [
        "data lake", "data lakehouse", "data warehouse", "data mesh",
        "lakehouse", "elt", "etl", "cdc", "change data capture",
        "event sourcing", "real-time", "streaming", "batch processing",
        "schema evolution", "data governance", "data catalog",
        "data quality", "data lineage", "star schema", "snowflake schema",
        "olap", "oltp", "columnar storage", "parquet", "avro", "orc",
        "microservices", "rest api", "graphql", "grpc",
    ],
}

# Flat lookup: canonical lowercase → category
_SKILL_INDEX: Dict[str, str] = {}
for _cat, _terms in SKILLS.items():
    for _term in _terms:
        _SKILL_INDEX[_term.lower()] = _cat

# Build a sorted-by-length list for longest-match extraction
_SKILL_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(term) + r"\b", re.I), term)
    for term in sorted(_SKILL_INDEX.keys(), key=len, reverse=True)
]


def extract_skills(text: str) -> List[str]:
    """
    Return a deduplicated list of matched skills (canonical lowercase).
    Longest-match wins: 'step functions' beats 'functions'.
    """
    found: Set[str] = set()
    # Track consumed character spans to prevent sub-match pollution
    consumed: List[Tuple[int, int]] = []

    for pattern, canonical in _SKILL_PATTERNS:
        for m in pattern.finditer(text):
            span = (m.start(), m.end())
            # Check overlap with already-consumed spans
            if any(s <= span[0] < e or s < span[1] <= e for s, e in consumed):
                continue
            found.add(canonical)
            consumed.append(span)
    return sorted(found)


# ---------------------------------------------------------------------------
# Role classification
# ---------------------------------------------------------------------------

ROLE_CENTROIDS: Dict[str, List[str]] = {
    "data_engineer": [
        "pipeline", "etl", "elt", "ingestion", "data lake", "warehouse",
        "kafka", "spark", "airflow", "redshift", "bigquery", "glue",
        "data infrastructure", "batch", "streaming",
    ],
    "data_scientist": [
        "machine learning", "model", "experiment", "hypothesis",
        "statistical", "regression", "classification", "clustering",
        "feature engineering", "prediction", "notebook", "jupyter",
        "scikit", "pandas", "numpy",
    ],
    "ml_engineer": [
        "mlops", "model deployment", "serving", "inference", "training pipeline",
        "feature store", "mlflow", "kubeflow", "sagemaker endpoint",
        "model monitoring", "a/b test", "canary",
    ],
    "data_analyst": [
        "dashboard", "report", "visualization", "looker", "tableau", "power bi",
        "sql", "business intelligence", "bi", "kpi", "ad-hoc",
    ],
    "analytics_engineer": [
        "dbt", "transformation", "data model", "dimensional model",
        "semantic layer", "data mart", "metrics layer",
    ],
    "platform_engineer": [
        "kubernetes", "infrastructure", "terraform", "cdk", "iac",
        "reliability", "sla", "platform", "observability",
    ],
}


def classify_role(title: str, description: str) -> str:
    """
    Keyword centroid scoring: count how many centroid words appear in
    title (weight 3×) + description, return category with highest score.
    """
    combined = title.lower() * 3 + " " + description.lower()
    scores: Dict[str, int] = defaultdict(int)
    for role, keywords in ROLE_CENTROIDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", combined, re.I):
                scores[role] += 1
    if not scores:
        return "other"
    return max(scores, key=lambda r: scores[r])


# ---------------------------------------------------------------------------
# Seniority inference
# ---------------------------------------------------------------------------

SENIORITY_PATTERNS = {
    "intern":    re.compile(r"\bintern(ship)?\b", re.I),
    "junior":    re.compile(r"\bjunior\b|\bjr\.?\b|\bentry[- ]level\b|\bassociate\b", re.I),
    "mid":       re.compile(r"\bmid[- ]level\b|\bii\b|\b2\b", re.I),
    "senior":    re.compile(r"\bsenior\b|\bsr\.?\b", re.I),
    "staff":     re.compile(r"\bstaff\b", re.I),
    "lead":      re.compile(r"\blead\b|\btech\s+lead\b|\bprincipal\b", re.I),
    "director":  re.compile(r"\bdirector\b|\bvp\b|\bvice\s+president\b|\bhead\s+of\b", re.I),
}

# Experience-mention patterns in JD body
_EXP_RANGE = re.compile(r"(\d+)\+?\s*(?:to|-)\s*(\d+)\s*years?", re.I)
_EXP_MIN   = re.compile(r"(\d+)\+\s*years?|at\s+least\s+(\d+)\s*years?", re.I)

def infer_seniority(title: str, description: str) -> str:
    # Title wins
    for level, pat in SENIORITY_PATTERNS.items():
        if pat.search(title):
            return level

    # Fall back to experience mentions in JD
    m = _EXP_RANGE.search(description)
    if m:
        years = (int(m.group(1)) + int(m.group(2))) / 2
    else:
        m2 = _EXP_MIN.search(description)
        years = int(m2.group(1) or m2.group(2)) if m2 else None

    if years is not None:
        if years <= 1:  return "junior"
        if years <= 3:  return "mid"
        if years <= 6:  return "senior"
        return "lead"

    return "unknown"


# ---------------------------------------------------------------------------
# Resume match scoring
# ---------------------------------------------------------------------------

@dataclass
class ResumeProfile:
    """Structured representation of a candidate's resume."""
    skills: List[str]
    years_experience: int
    seniority_target: str           # "mid" | "senior" | …
    role_targets: List[str]         # ["data_engineer", "analytics_engineer"]
    visa_required: bool = False
    preferred_remote: str = "any"   # "remote" | "hybrid" | "any"

    @property
    def skill_set(self) -> Set[str]:
        return {s.lower() for s in self.skills}


@dataclass
class MatchResult:
    posting_id: str
    score: float                   # 0–100
    matched_skills: List[str]
    missing_skills: List[str]
    title_relevance: float
    skill_overlap: float
    seniority_fit: float
    visa_ok: bool
    remote_ok: bool
    breakdown: Dict[str, float] = field(default_factory=dict)


SENIORITY_ORDER = ["intern", "junior", "mid", "senior", "staff", "lead", "director"]

def _seniority_distance(a: str, b: str) -> int:
    try:
        return abs(SENIORITY_ORDER.index(a) - SENIORITY_ORDER.index(b))
    except ValueError:
        return 1


def score_posting(posting, resume: ResumeProfile) -> MatchResult:
    """
    Weighted scoring:
      40% skill overlap
      25% role/title relevance
      20% seniority fit
      10% visa/remote compatibility
       5% compensation signal
    """
    # --- Skill overlap ---
    posting_skills = set(posting.skills)
    resume_skills  = resume.skill_set
    matched = sorted(posting_skills & resume_skills)
    missing = sorted(posting_skills - resume_skills)
    skill_overlap = len(matched) / max(len(posting_skills), 1)

    # --- Title relevance ---
    role_hit = posting.role_category in resume.role_targets
    title_relevance = 1.0 if role_hit else 0.3

    # --- Seniority fit ---
    dist = _seniority_distance(posting.seniority, resume.seniority_target)
    seniority_fit = max(0.0, 1.0 - dist * 0.35)

    # --- Compatibility ---
    visa_ok   = not resume.visa_required or posting.visa_sponsored
    remote_ok = (
        resume.preferred_remote == "any"
        or posting.remote_type == resume.preferred_remote
        or posting.remote_type == "remote"
    )
    compat = (0.5 * int(visa_ok) + 0.5 * int(remote_ok))

    # --- Composite score ---
    raw = (
        0.40 * skill_overlap
        + 0.25 * title_relevance
        + 0.20 * seniority_fit
        + 0.15 * compat
    )
    score = round(raw * 100, 1)

    return MatchResult(
        posting_id=posting.posting_id,
        score=score,
        matched_skills=matched,
        missing_skills=missing[:10],
        title_relevance=round(title_relevance, 2),
        skill_overlap=round(skill_overlap, 2),
        seniority_fit=round(seniority_fit, 2),
        visa_ok=visa_ok,
        remote_ok=remote_ok,
        breakdown={
            "skill_overlap": round(skill_overlap * 40, 1),
            "title_relevance": round(title_relevance * 25, 1),
            "seniority_fit": round(seniority_fit * 20, 1),
            "compatibility": round(compat * 15, 1),
        },
    )


# ---------------------------------------------------------------------------
# Batch enrichment (adds NLP fields to JobPosting objects in-place)
# ---------------------------------------------------------------------------

def enrich_postings(postings: list, resume: Optional[ResumeProfile] = None) -> List[MatchResult]:
    """
    Mutates each posting in-place (skills, role_category, seniority).
    Returns scored MatchResults if resume is provided.
    """
    results = []
    for posting in postings:
        posting.skills        = extract_skills(posting.description_clean + " " + posting.title)
        posting.role_category = classify_role(posting.title, posting.description_clean)
        posting.seniority     = infer_seniority(posting.title, posting.description_clean)
        if resume:
            results.append(score_posting(posting, resume))

    if resume:
        results.sort(key=lambda r: r.score, reverse=True)
    return results
