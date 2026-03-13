"""
job_collector.py  (updated — uses misceres/indeed-scraper which returns HTTP 201)
"""

import hashlib
import html
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlencode
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------

@dataclass
class JobPosting:
    posting_id: str
    external_id: str
    source: str
    title: str
    company: str
    location: str
    remote_type: str
    employment_type: str
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    salary_currency: str = "USD"
    salary_period: str = "year"
    description_raw: str = ""
    description_clean: str = ""
    skills: List[str] = field(default_factory=list)
    role_category: str = "unknown"
    seniority: str = "unknown"
    visa_sponsored: bool = False
    posted_at: Optional[datetime] = None
    collected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    apply_url: str = ""
    raw_payload: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["posted_at"]    = self.posted_at.isoformat()    if self.posted_at    else None
        d["collected_at"] = self.collected_at.isoformat() if self.collected_at else None
        return d


def make_posting_id(source: str, external_id: str) -> str:
    raw = f"{source}::{external_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------

def _build_session(max_retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "JobIntelligencePipeline/1.0"})
    return session


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_DATE_FMTS = [
    "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y",
]

def parse_date(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    s = str(raw).strip()
    m = re.match(r"(\d+)\s+day", s, re.I)
    if m:
        from datetime import timedelta
        return datetime.now(timezone.utc) - timedelta(days=int(m.group(1)))
    # Handle "Just posted" and similar
    if re.search(r"just\s+posted|today|now", s, re.I):
        return datetime.now(timezone.utc)
    for fmt in _DATE_FMTS:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except ValueError:
            continue
    return None


def clean_html(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_SALARY_RE = re.compile(
    r"""(?:up\s+to\s+)?[$£€]?\s*(?P<lo>[\d,\.]+)\s*[kK]?
    (?:\s*[-–—to]+\s*[$£€]?\s*(?P<hi>[\d,\.]+)\s*[kK]?)?
    (?:\s+(?P<period>per\s+(?:year|annum|hour|month)|\bpa\b|\byr\b|\bhr\b))?""",
    re.VERBOSE | re.IGNORECASE,
)
_CURRENCY_SYMBOLS = {"$": "USD", "£": "GBP", "€": "EUR"}

def parse_salary(text: str) -> Tuple[Optional[float], Optional[float], str, str]:
    if not text:
        return None, None, "USD", "year"
    currency = "USD"
    for sym, code in _CURRENCY_SYMBOLS.items():
        if sym in text:
            currency = code
            break
    m = _SALARY_RE.search(text)
    if not m:
        return None, None, currency, "year"

    def to_float(s):
        if not s: return None
        v = float(s.replace(",", ""))
        if re.search(r"[kK]", text[m.start():m.end()]):
            v *= 1000
        return v

    lo = to_float(m.group("lo"))
    hi = to_float(m.group("hi"))
    period_raw = (m.group("period") or "").lower()
    period = "hour" if re.search(r"hour|hr", period_raw) else "month" if "month" in period_raw else "year"
    return lo, hi, currency, period


_REMOTE_PATTERNS = {
    "remote":  re.compile(r"\bremote\b", re.I),
    "hybrid":  re.compile(r"\bhybrid\b", re.I),
    "onsite":  re.compile(r"\bon.?site\b|\bin.?office\b|\bin.?person\b", re.I),
}

def infer_remote_type(text: str) -> str:
    for label, pat in _REMOTE_PATTERNS.items():
        if pat.search(text):
            return label
    return "unknown"


_EMP_PATTERNS = {
    "full_time": re.compile(r"\bfull[- ]?time\b|\bft\b", re.I),
    "part_time": re.compile(r"\bpart[- ]?time\b|\bpt\b", re.I),
    "contract":  re.compile(r"\bcontract\b|\bfreelance\b|\btemporary\b|\btemp\b", re.I),
}

def infer_employment_type(text: str) -> str:
    for label, pat in _EMP_PATTERNS.items():
        if pat.search(text):
            return label
    return "unknown"


def infer_visa(text: str) -> bool:
    return bool(re.search(r"visa\s+sponsor|h1.?b\s+sponsor|work\s+authori[sz]", text, re.I))


# ---------------------------------------------------------------------------
# LinkedIn Adapter (unchanged)
# ---------------------------------------------------------------------------

class LinkedInAdapter:
    SOURCE = "linkedin"

    def __init__(self, api_key: str, base_url: str = "https://linkedin-jobs-search.p.rapidapi.com"):
        self.session = _build_session()
        self.session.headers.update({
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": base_url.split("//")[-1],
        })
        self.base_url = base_url

    def fetch(self, query: str, location: str = "United States", count: int = 50) -> List[Dict]:
        url = f"{self.base_url}/"
        params = {"keywords": query, "locationId": location, "count": str(count)}
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json() if isinstance(resp.json(), list) else resp.json().get("elements", [])

    def normalise(self, raw: Dict) -> JobPosting:
        title = raw.get("title", "")
        company = (raw.get("companyDetails") or {}).get("name") or raw.get("company", "")
        loc_data = raw.get("formattedLocation") or raw.get("location", "")
        desc = raw.get("description", {})
        desc_text = desc.get("text") if isinstance(desc, dict) else str(desc)
        salary_text = raw.get("salary") or raw.get("compensationDescription", "")
        remote_flag = raw.get("workRemoteAllowed", False)
        smin, smax, scur, sper = parse_salary(salary_text)
        description_clean = clean_html(desc_text)
        return JobPosting(
            posting_id=make_posting_id(self.SOURCE, str(raw.get("entityUrn", raw.get("id", "")))),
            external_id=str(raw.get("entityUrn", raw.get("id", ""))),
            source=self.SOURCE, title=title, company=company, location=str(loc_data),
            remote_type="remote" if remote_flag else infer_remote_type(title + " " + str(loc_data)),
            employment_type=infer_employment_type(raw.get("employmentType", "")),
            salary_min=smin, salary_max=smax, salary_currency=scur, salary_period=sper,
            description_raw=desc_text or "", description_clean=description_clean,
            visa_sponsored=infer_visa(description_clean),
            posted_at=parse_date(raw.get("listedAt") or raw.get("postAt")),
            apply_url=raw.get("applyUrl", ""), raw_payload=raw,
        )

    def collect(self, query: str, location: str = "United States", count: int = 50) -> List[JobPosting]:
        raw_list = self.fetch(query, location, count)
        postings = []
        for item in raw_list:
            try:
                postings.append(self.normalise(item))
            except Exception as exc:
                logger.warning("LinkedIn normalise error: %s", exc)
        logger.info("LinkedIn: collected %d/%d postings", len(postings), len(raw_list))
        return postings


# ---------------------------------------------------------------------------
# Indeed Adapter  — uses misceres/indeed-scraper (returns HTTP 200 or 201)
# ---------------------------------------------------------------------------

class IndeedAdapter:
    """
    Uses the Apify misceres/indeed-scraper actor.

    Key fix: Apify returns HTTP 201 (not 200) when a run is created and
    results are returned inline. Both 200 and 201 are success responses.

    Field mapping for misceres actor:
        positionName  → job title
        company       → company name
        location      → location string
        salary        → raw salary text  (e.g. "$23 an hour", "$80,000-$100,000")
        postedAt      → posted date      (e.g. "Just posted", "2 days ago")
        externalApplyLink / jobUrl → apply URL
        jobType       → employment type
    """
    SOURCE = "indeed"

    def __init__(self, apify_token: str):
        self.session   = _build_session()
        self.apify_token = apify_token
        self.actor_url = (
            "https://api.apify.com/v2/acts/misceres~indeed-scraper"
            "/run-sync-get-dataset-items"
        )

    def fetch(self, position: str, location: str = "United States", max_items: int = 50) -> List[Dict]:
        logger.info("Indeed fetch | position=%r location=%r max_items=%d", position, location, max_items)
        payload = {
            "keyword":  position,    # misceres uses "keyword"
            "location": location,
            "maxItems": max_items,
        }
        resp = self.session.post(
            self.actor_url,
            json=payload,
            params={"token": self.apify_token},
            timeout=180,
        )

        # 200 and 201 are BOTH success for Apify run-sync endpoint
        if resp.status_code not in (200, 201):
            logger.error("Indeed scraper failed | status=%d | body=%s",
                         resp.status_code, resp.text[:300])
            return []

        data = resp.json()
        if not isinstance(data, list):
            logger.warning("Unexpected response type: %s", type(data))
            return []

        logger.info("Indeed API returned %d raw items", len(data))
        return data

    @staticmethod
    def _to_str(value) -> str:
        """
        Safely convert any API field to a plain string.
        The misceres actor sometimes returns fields as lists, e.g.:
            jobType -> ["Full-time", "Contract"]
            salary  -> ["$23 an hour"]
        This ensures string concatenation never crashes.
        """
        if value is None:
            return ""
        if isinstance(value, list):
            return " ".join(str(v) for v in value if v)
        return str(value)

    def normalise(self, raw: Dict) -> JobPosting:
        # _to_str() safely handles fields that may be lists or None
        title       = self._to_str(raw.get("positionName") or raw.get("title"))
        company     = self._to_str(raw.get("company"))
        location    = self._to_str(raw.get("location"))
        salary_text = self._to_str(raw.get("salary"))
        job_type    = self._to_str(raw.get("jobType"))

        # Description — prefer plain text over HTML
        desc = self._to_str(
            raw.get("description")
            or raw.get("jobDescription")
            or raw.get("summary")
        )

        # Apply URL — try multiple field names
        apply_url = self._to_str(
            raw.get("externalApplyLink")
            or raw.get("applyLink")
            or raw.get("jobUrl")
            or raw.get("url")
        )

        # Stable unique ID
        external_id = (
            self._to_str(raw.get("jobId") or raw.get("id") or raw.get("jobKey"))
            or f"{title[:20]}_{company[:10]}"
        )

        smin, smax, scur, sper = parse_salary(salary_text)
        description_clean = clean_html(desc)

        return JobPosting(
            posting_id=make_posting_id(self.SOURCE, external_id),
            external_id=external_id,
            source=self.SOURCE,
            title=title,
            company=company,
            location=location,
            remote_type=infer_remote_type(title + " " + location + " " + description_clean[:500]),
            employment_type=infer_employment_type(job_type + " " + description_clean[:300]),
            salary_min=smin,
            salary_max=smax,
            salary_currency=scur,
            salary_period=sper,
            description_raw=desc,
            description_clean=description_clean,
            visa_sponsored=infer_visa(description_clean),
            posted_at=parse_date(raw.get("postedAt") or raw.get("date")),
            apply_url=apply_url,
            raw_payload=raw,
        )

    def collect(self, position: str, location: str = "United States", max_items: int = 50) -> List[JobPosting]:
        raw_list = self.fetch(position, location, max_items)
        postings = []
        for item in raw_list:
            try:
                postings.append(self.normalise(item))
            except Exception as exc:
                logger.warning("Indeed normalise error: %s | keys=%s", exc, list(item.keys()))
        logger.info("Indeed: collected %d/%d postings", len(postings), len(raw_list))
        return postings


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class DeduplicationStore:
    def __init__(self):
        self._seen: set = set()

    def is_new(self, posting: JobPosting) -> bool:
        if posting.posting_id in self._seen:
            return False
        self._seen.add(posting.posting_id)
        return True

    def filter_new(self, postings: List[JobPosting]) -> List[JobPosting]:
        new = [p for p in postings if self.is_new(p)]
        dupes = len(postings) - len(new)
        if dupes:
            logger.info("Deduplicated %d duplicate postings", dupes)
        return new

    @property
    def size(self) -> int:
        return len(self._seen)