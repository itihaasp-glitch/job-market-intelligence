"""
pipeline.py
-----------
Top-level orchestrator for the Job Intelligence Pipeline.

Wires together: collect → deduplicate → enrich → score → store → archive

Usage
=====
python pipeline.py \
    --queries "Data Engineer" "Senior Data Engineer" \
    --locations "New York" "Remote" \
    --resume resume.txt \
    --output-html report.html \
    --backend sqlite \
    --db-path jobs.db
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Local modules
sys.path.insert(0, str(Path(__file__).parent))
from job_collector import LinkedInAdapter, IndeedAdapter, DeduplicationStore, JobPosting
from nlp_extractor import enrich_postings, extract_skills, ResumeProfile
from storage import get_store, S3Archive

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Resume parsing (plain-text)
# ---------------------------------------------------------------------------

def parse_resume_txt(path: str) -> ResumeProfile:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    skills = extract_skills(text)

    import re
    # Years of experience
    m = re.search(r"(\d+)\+?\s*years?\s+(?:of\s+)?experience", text, re.I)
    years = int(m.group(1)) if m else 3

    # Seniority from resume header
    seniority = "mid"
    for level in ["senior", "staff", "lead", "junior", "director", "principal"]:
        if re.search(r"\b" + level + r"\b", text[:500], re.I):
            seniority = level
            break

    visa = bool(re.search(r"authorized\s+to\s+work|visa\s+sponsor|h1.?b", text, re.I))

    return ResumeProfile(
        skills=skills,
        years_experience=years,
        seniority_target=seniority,
        role_targets=["data_engineer", "analytics_engineer"],
        visa_required=not visa,  # if not authorised, needs sponsorship
        preferred_remote="any",
    )


# ---------------------------------------------------------------------------
# HTML report generator
# ---------------------------------------------------------------------------

def generate_html_report(
    postings: List[JobPosting],
    match_results: list,
    output_path: str,
    resume_profile: Optional[ResumeProfile] = None,
) -> None:
    """Generate a self-contained HTML report of top job matches."""

    score_map = {r.posting_id: r for r in match_results}

    rows_html = ""
    for posting in postings[:100]:
        match = score_map.get(posting.posting_id)
        score = match.score if match else "—"
        score_color = (
            "#22c55e" if isinstance(score, float) and score >= 70 else
            "#f59e0b" if isinstance(score, float) and score >= 50 else
            "#ef4444"
        )
        skills_html = " ".join(
            f'<span class="skill-tag">{s}</span>' for s in posting.skills[:8]
        )
        salary = "—"
        if posting.salary_min:
            salary = f"${posting.salary_min:,.0f}"
            if posting.salary_max:
                salary += f" – ${posting.salary_max:,.0f}"
            salary += f"/{posting.salary_period[:2]}"

        rows_html += f"""
        <tr>
          <td>
            <div class="job-title">{posting.title}</div>
            <div class="company">{posting.company}</div>
          </td>
          <td>{posting.location}</td>
          <td><span class="badge badge-{posting.remote_type}">{posting.remote_type}</span></td>
          <td><span class="badge badge-{posting.seniority}">{posting.seniority}</span></td>
          <td class="salary">{salary}</td>
          <td>{skills_html}</td>
          <td><span class="score" style="color:{score_color}">{score}</span></td>
          <td><a href="{posting.apply_url}" target="_blank" class="apply-btn">Apply →</a></td>
        </tr>"""

    collected_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Job Intelligence Report — {collected_at}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }}
  h1 {{ font-size: 1.8rem; color: #38bdf8; margin-bottom: 0.25rem; }}
  .subtitle {{ color: #94a3b8; margin-bottom: 2rem; font-size: 0.9rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ background: #1e293b; color: #94a3b8; text-align: left; padding: 0.75rem 1rem;
        font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; font-size: 0.75rem; }}
  tr:nth-child(even) {{ background: #1e293b55; }}
  tr:hover {{ background: #1e293b; }}
  td {{ padding: 0.7rem 1rem; vertical-align: top; border-bottom: 1px solid #1e293b; }}
  .job-title {{ font-weight: 600; color: #f1f5f9; }}
  .company {{ color: #94a3b8; font-size: 0.8rem; margin-top: 2px; }}
  .salary {{ color: #4ade80; font-weight: 500; white-space: nowrap; }}
  .score {{ font-weight: 700; font-size: 1rem; }}
  .skill-tag {{ display: inline-block; background: #0f3460; color: #38bdf8;
                padding: 2px 6px; border-radius: 4px; font-size: 0.72rem; margin: 2px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 99px; font-size: 0.72rem; font-weight: 500; }}
  .badge-remote {{ background: #064e3b; color: #34d399; }}
  .badge-hybrid {{ background: #1e3a5f; color: #60a5fa; }}
  .badge-onsite {{ background: #3b1f1f; color: #f87171; }}
  .badge-unknown {{ background: #2d2d2d; color: #9ca3af; }}
  .badge-senior {{ background: #312e81; color: #a78bfa; }}
  .badge-mid    {{ background: #1c3a2a; color: #34d399; }}
  .badge-junior {{ background: #2d1e0e; color: #fb923c; }}
  .badge-lead   {{ background: #1e293b; color: #e2e8f0; }}
  .apply-btn {{ color: #38bdf8; text-decoration: none; font-weight: 600; white-space: nowrap; }}
  .apply-btn:hover {{ color: #7dd3fc; }}
  .stats {{ display: flex; gap: 1.5rem; margin-bottom: 2rem; flex-wrap: wrap; }}
  .stat {{ background: #1e293b; border-radius: 8px; padding: 1rem 1.5rem; min-width: 140px; }}
  .stat-val {{ font-size: 1.6rem; font-weight: 700; color: #38bdf8; }}
  .stat-lbl {{ color: #94a3b8; font-size: 0.8rem; }}
</style>
</head>
<body>
<h1>🔍 Job Intelligence Report</h1>
<p class="subtitle">Collected {collected_at} · {len(postings)} postings · ranked by match score</p>

<div class="stats">
  <div class="stat"><div class="stat-val">{len(postings)}</div><div class="stat-lbl">Postings collected</div></div>
  <div class="stat"><div class="stat-val">{sum(1 for p in postings if p.visa_sponsored)}</div><div class="stat-lbl">Visa sponsored</div></div>
  <div class="stat"><div class="stat-val">{sum(1 for p in postings if p.remote_type == 'remote')}</div><div class="stat-lbl">Remote roles</div></div>
  <div class="stat"><div class="stat-val">{len([r for r in match_results if r.score >= 70])}</div><div class="stat-lbl">Strong matches (≥70)</div></div>
</div>

<table>
<thead>
  <tr>
    <th>Role</th><th>Location</th><th>Remote</th><th>Seniority</th>
    <th>Salary</th><th>Skills</th><th>Score</th><th>Apply</th>
  </tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    logger.info("HTML report written → %s", output_path)


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def run_pipeline(
    queries: List[str],
    locations: List[str],
    linkedin_key: Optional[str] = None,
    apify_token: Optional[str]  = None,
    resume_path: Optional[str]  = None,
    output_html: str            = "report.html",
    backend: str                = "sqlite",
    db_path: str                = "jobs.db",
    s3_bucket: Optional[str]    = None,
) -> dict:

    store  = get_store(backend, db_path=db_path)
    dedup  = DeduplicationStore()
    all_postings: List[JobPosting] = []

    # ── Collect ──────────────────────────────────────────────────────────────
    for query in queries:
        for location in locations:
            if linkedin_key:
                try:
                    adapter  = LinkedInAdapter(api_key=linkedin_key)
                    postings = adapter.collect(query, location, count=50)
                    all_postings.extend(postings)
                except Exception as exc:
                    logger.warning("LinkedIn collect failed: %s", exc)

            if apify_token:
                try:
                    adapter  = IndeedAdapter(apify_token=apify_token)
                    postings = adapter.collect(query, location, max_items=50)
                    all_postings.extend(postings)
                except Exception as exc:
                    logger.warning("Indeed collect failed: %s", exc)

    # ── Deduplicate ───────────────────────────────────────────────────────────
    all_postings = dedup.filter_new(all_postings)
    logger.info("After dedup: %d unique postings", len(all_postings))

    # ── NLP enrichment + scoring ──────────────────────────────────────────────
    resume_profile = None
    match_results  = []
    resume_hash    = ""

    if resume_path and Path(resume_path).exists():
        resume_profile = parse_resume_txt(resume_path)
        resume_text    = Path(resume_path).read_text(encoding="utf-8", errors="replace")
        resume_hash    = hashlib.sha256(resume_text.encode()).hexdigest()[:16]
        logger.info("Resume loaded | skills=%d | seniority=%s",
                    len(resume_profile.skills), resume_profile.seniority_target)

    match_results = enrich_postings(all_postings, resume=resume_profile)

    # Sort postings by match score (if scored), else by collected_at
    if match_results:
        score_map = {r.posting_id: r.score for r in match_results}
        all_postings.sort(key=lambda p: score_map.get(p.posting_id, 0), reverse=True)

    # ── Persist ───────────────────────────────────────────────────────────────
    if all_postings:
        store.upsert_postings(all_postings)
    if match_results and hasattr(store, "save_match_results"):
        store.save_match_results(match_results, resume_hash)

    # ── Archive to S3 ─────────────────────────────────────────────────────────
    s3_key = ""
    if s3_bucket and all_postings:
        archive = S3Archive(bucket=s3_bucket)
        s3_key  = archive.archive_raw(all_postings)

    # ── Report ────────────────────────────────────────────────────────────────
    generate_html_report(all_postings, match_results, output_html, resume_profile)

    summary = {
        "postings_collected": len(all_postings),
        "sources": list({p.source for p in all_postings}),
        "visa_sponsored": sum(1 for p in all_postings if p.visa_sponsored),
        "remote_roles": sum(1 for p in all_postings if p.remote_type == "remote"),
        "strong_matches": len([r for r in match_results if r.score >= 70]),
        "top_matches": [
            {"posting_id": r.posting_id, "score": r.score,
             "matched_skills": r.matched_skills[:5]}
            for r in match_results[:5]
        ],
        "html_report": output_html,
        "s3_archive": s3_key,
    }
    logger.info("Pipeline complete: %s", json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Job Intelligence Pipeline")
    parser.add_argument("--queries",   nargs="+", default=["Data Engineer"],
                        help="Job search queries")
    parser.add_argument("--locations", nargs="+", default=["United States"],
                        help="Search locations")
    parser.add_argument("--resume",    default=None, help="Path to plain-text resume")
    parser.add_argument("--output-html", default="report.html")
    parser.add_argument("--backend",   choices=["sqlite", "postgres"], default="sqlite")
    parser.add_argument("--db-path",   default="jobs.db")
    parser.add_argument("--s3-bucket", default=None)
    args = parser.parse_args()

    result = run_pipeline(
        queries       = args.queries,
        locations     = args.locations,
        linkedin_key  = os.getenv("LINKEDIN_API_KEY"),
        apify_token   = os.getenv("APIFY_TOKEN"),
        resume_path   = args.resume,
        output_html   = args.output_html,
        backend       = args.backend,
        db_path       = args.db_path,
        s3_bucket     = args.s3_bucket,
    )
    print(json.dumps(result, indent=2))
