#!/usr/bin/env python3
"""
build.py - Static site generator for AI Legal News aggregator.

Fetches RSS feeds, filters and deduplicates articles, and renders
static HTML using Jinja2 templates. Output goes to docs/ for GitHub Pages.
"""

import json
import hashlib
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import feedparser
import requests
from jinja2 import Environment, FileSystemLoader
import bleach

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
TEMPLATES_DIR = ROOT_DIR / "templates"
STATIC_DIR = ROOT_DIR / "static"
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "docs"
FEEDS_FILE = DATA_DIR / "feeds.json"
SEEN_FILE = DATA_DIR / "seen_articles.json"

BASE_URL = "/AI-Legal-News"  # GitHub Pages subdirectory path (no trailing slash)

MAX_ARTICLE_AGE_DAYS = 90
MAX_ARTICLES_HOMEPAGE = 80
MAX_SUMMARY_LENGTH = 300
REQUEST_TIMEOUT = 30
MAX_WORKERS = 8
MAX_SEEN_ENTRIES = 5000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category definitions (order determines nav display)
# ---------------------------------------------------------------------------

CATEGORIES = {
    "regulation": {
        "label": "US Regulators",
        "description": "Federal regulation, FTC actions, NIST standards, executive orders",
    },
    "state-regulation": {
        "label": "State AGs",
        "description": "State attorney general actions and enforcement",
    },
    "international": {
        "label": "International",
        "description": "EU AI Act, GDPR, UK ICO, global AI governance",
    },
    "law-firms": {
        "label": "Law Firms",
        "description": "Commentary and analysis from major law firms",
    },
    "legal-analysis": {
        "label": "Legal Analysis",
        "description": "Legal publications and scholarly analysis",
    },
    "legal-tech": {
        "label": "Legal Tech",
        "description": "AI tools for lawyers, legal technology innovation",
    },
    "policy": {
        "label": "Policy",
        "description": "Think tank analysis, civil liberties, policy proposals",
    },
    "research-policy": {
        "label": "Research",
        "description": "Academic research on AI policy and governance",
    },
    "tech-news": {
        "label": "Tech News",
        "description": "Technology news with AI legal implications",
    },
    "litigation": {
        "label": "Litigation",
        "description": "Lawsuits, court rulings, legal disputes involving AI",
    },
    "copyright-ip": {
        "label": "Copyright & IP",
        "description": "AI and intellectual property, training data rights",
    },
    "ethics": {
        "label": "Ethics & Governance",
        "description": "AI ethics frameworks, responsible AI, bias",
    },
}

# ---------------------------------------------------------------------------
# Keyword sets (referenced by shorthand in feeds.json)
# ---------------------------------------------------------------------------

AI_KEYWORDS = [
    "ai", "artificial intelligence", "algorithm", "automated",
    "machine learning", "deepfake", "generative ai",
    "large language model", "llm", "neural network",
    "chatbot", "gpt", "openai", "anthropic", "copilot",
]

LEGAL_KEYWORDS = [
    "law", "legal", "regulation", "sued", "lawsuit", "court",
    "ruling", "legislation", "ftc", "copyright", "patent",
    "policy", "compliance", "enforcement", "attorney general",
    "settlement", "privacy", "governance",
]


def _resolve_keywords(feed_config: dict, config: dict) -> list[str]:
    """Resolve keyword shorthand ('ai', 'legal', []) to full keyword list."""
    kw = feed_config.get("keywords", [])
    if isinstance(kw, str):
        if kw == "ai":
            return config.get("ai_keywords", AI_KEYWORDS)
        elif kw == "legal":
            return config.get("legal_keywords", LEGAL_KEYWORDS)
        else:
            return [kw]
    return kw

# ---------------------------------------------------------------------------
# Feed fetching
# ---------------------------------------------------------------------------


def load_feed_config() -> dict:
    """Load feed configuration from feeds.json."""
    with open(FEEDS_FILE, "r") as f:
        return json.load(f)


def fetch_single_feed(feed_config: dict) -> list[dict]:
    """Fetch and parse a single RSS feed. Returns normalised article dicts."""
    url = feed_config["url"]
    name = feed_config["name"]
    keywords = feed_config.get("_resolved_keywords", feed_config.get("keywords", []))
    if isinstance(keywords, str):
        keywords = []  # safety fallback
    category = feed_config.get("category", "uncategorized")

    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={
                "User-Agent": "AILegalNewsBot/1.0 (+https://github.com/ai-legal-news)"
            },
        )
        response.raise_for_status()
        feed = feedparser.parse(response.content)
    except requests.RequestException as e:
        logger.warning("Failed to fetch feed '%s' (%s): %s", name, url, e)
        return []
    except Exception as e:
        logger.warning("Failed to parse feed '%s': %s", name, e)
        return []

    if feed.bozo and not feed.entries:
        logger.warning("Feed '%s' is malformed and has no entries.", name)
        return []

    articles = []
    for entry in feed.entries:
        article = _normalize_entry(entry, name, category)
        if article is None:
            continue
        if keywords and not _matches_keywords(article, keywords):
            continue
        articles.append(article)

    logger.info("Fetched %d articles from '%s'", len(articles), name)
    return articles


def fetch_all_feeds(feed_configs: list[dict]) -> list[dict]:
    """Fetch all feeds concurrently."""
    all_articles: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_feed = {
            executor.submit(fetch_single_feed, fc): fc for fc in feed_configs
        }
        for future in as_completed(future_to_feed):
            feed_conf = future_to_feed[future]
            try:
                all_articles.extend(future.result())
            except Exception as e:
                logger.error(
                    "Unexpected error fetching '%s': %s", feed_conf["name"], e
                )
    return all_articles


# ---------------------------------------------------------------------------
# Article normalisation helpers
# ---------------------------------------------------------------------------


def _normalize_entry(entry, source_name: str, category: str) -> dict | None:
    """Normalise a feedparser entry into a consistent dict."""
    title = entry.get("title", "").strip()
    link = entry.get("link", "").strip()
    if not title or not link:
        return None

    published = _parse_date(entry)
    summary = _extract_summary(entry)
    article_id = hashlib.sha256(link.lower().strip().encode()).hexdigest()[:16]

    return {
        "id": article_id,
        "title": title,
        "link": link,
        "source": source_name,
        "category": category,
        "published": published,
        "published_str": published.strftime("%B %d, %Y"),
        "published_iso": published.isoformat(),
        "summary": summary,
    }


def _parse_date(entry) -> datetime:
    """Extract a datetime from a feed entry, falling back to now."""
    for field in ("published_parsed", "updated_parsed", "created_parsed"):
        ts = entry.get(field)
        if ts:
            try:
                return datetime(*ts[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
    return datetime.now(timezone.utc)


def _extract_summary(entry) -> str:
    """Extract a clean text summary from a feed entry."""
    raw = ""
    if entry.get("summary"):
        raw = entry["summary"]
    elif entry.get("content"):
        raw = entry["content"][0].get("value", "")
    elif entry.get("description"):
        raw = entry["description"]

    if not raw:
        return "No summary available."

    clean = bleach.clean(raw, tags=[], strip=True).strip()
    clean = " ".join(clean.split())

    if len(clean) > MAX_SUMMARY_LENGTH:
        clean = clean[:MAX_SUMMARY_LENGTH].rsplit(" ", 1)[0] + "\u2026"

    return clean if clean else "No summary available."


def _matches_keywords(article: dict, keywords: list[str]) -> bool:
    """Check if an article matches any required keywords."""
    text = (article["title"] + " " + article["summary"]).lower()
    return any(kw.lower() in text for kw in keywords)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def load_seen_articles() -> dict:
    if SEEN_FILE.exists():
        try:
            with open(SEEN_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Could not load seen_articles.json: %s", e)
    return {}


def save_seen_articles(seen: dict) -> None:
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=MAX_ARTICLE_AGE_DAYS)
    ).isoformat()
    pruned = {aid: ts for aid, ts in seen.items() if ts >= cutoff}
    if len(pruned) > MAX_SEEN_ENTRIES:
        pruned = dict(
            sorted(pruned.items(), key=lambda x: x[1], reverse=True)[:MAX_SEEN_ENTRIES]
        )
    with open(SEEN_FILE, "w") as f:
        json.dump(pruned, f, indent=2)


def deduplicate(articles: list[dict]) -> list[dict]:
    """Remove cross-feed duplicates within the current batch."""
    seen_ids: set[str] = set()
    unique: list[dict] = []
    for article in articles:
        if article["id"] not in seen_ids:
            seen_ids.add(article["id"])
            unique.append(article)
    return unique


# ---------------------------------------------------------------------------
# Categorisation
# ---------------------------------------------------------------------------

_CATEGORY_PATTERNS: list[tuple[str, list[str]]] = [
    (
        "litigation",
        [
            "lawsuit", "sued", "court ruling", "plaintiff", "defendant",
            "settlement", "injunction", "class action", "damages",
            "filed suit", "jury", "verdict",
        ],
    ),
    (
        "copyright-ip",
        [
            "copyright", "intellectual property", "patent", "trademark",
            "training data", "fair use", "dmca", "creative commons",
        ],
    ),
    (
        "ethics",
        [
            "ethics", "bias", "fairness", "transparency", "accountability",
            "responsible ai", "algorithmic", "discrimination",
        ],
    ),
]


def _auto_categorize(article: dict) -> str:
    """Refine category based on article content keywords."""
    text = (article["title"] + " " + article["summary"]).lower()
    for cat, keywords in _CATEGORY_PATTERNS:
        if any(kw in text for kw in keywords):
            return cat
    return article["category"]


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------


def _group_by_date(articles: list[dict]) -> list[tuple[str, list[dict]]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for article in articles:
        grouped[article["published_str"]].append(article)
    return sorted(
        grouped.items(),
        key=lambda x: x[1][0]["published"],
        reverse=True,
    )


def _group_by_category(articles: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for article in articles:
        grouped[article["category"]].append(article)
    return dict(grouped)


# ---------------------------------------------------------------------------
# Site generation
# ---------------------------------------------------------------------------


def build_site(articles: list[dict]) -> None:
    """Render all static HTML pages and write to docs/."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)

    # Apply auto-categorisation
    for article in articles:
        article["category"] = _auto_categorize(article)

    # Sort newest first
    articles.sort(key=lambda a: a["published"], reverse=True)

    homepage_articles = articles[:MAX_ARTICLES_HOMEPAGE]
    date_groups = _group_by_date(homepage_articles)
    category_groups = _group_by_category(articles)
    build_time = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")
    source_count = len({a["source"] for a in articles})
    category_counts = {k: len(v) for k, v in category_groups.items()}

    # Ensure output dirs exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "category").mkdir(exist_ok=True)
    (OUTPUT_DIR / "css").mkdir(exist_ok=True)
    (OUTPUT_DIR / "img").mkdir(exist_ok=True)

    shared = dict(
        base_url=BASE_URL,
        categories=CATEGORIES,
        category_counts=category_counts,
        build_time=build_time,
        total_articles=len(articles),
        source_count=source_count,
    )

    # Homepage
    html = env.get_template("index.html").render(
        date_groups=date_groups, active_page="home", **shared
    )
    (OUTPUT_DIR / "index.html").write_text(html, encoding="utf-8")

    # Archive
    html = env.get_template("archive.html").render(
        date_groups=_group_by_date(articles), active_page="archive", **shared
    )
    (OUTPUT_DIR / "archive.html").write_text(html, encoding="utf-8")

    # Category pages
    cat_tmpl = env.get_template("category.html")
    for cat_key, cat_articles in category_groups.items():
        cat_info = CATEGORIES.get(
            cat_key,
            {"label": cat_key.replace("-", " ").title(), "description": ""},
        )
        html = cat_tmpl.render(
            category=cat_info,
            category_key=cat_key,
            date_groups=_group_by_date(cat_articles),
            active_page=cat_key,
            total_articles=len(cat_articles),
            categories=CATEGORIES,
            category_counts=category_counts,
            build_time=build_time,
            source_count=source_count,
        )
        (OUTPUT_DIR / "category" / f"{cat_key}.html").write_text(html, encoding="utf-8")

    # Copy static assets
    css_src = STATIC_DIR / "css" / "style.css"
    if css_src.exists():
        shutil.copy2(css_src, OUTPUT_DIR / "css" / "style.css")
    img_src = STATIC_DIR / "img"
    if img_src.exists():
        for img_file in img_src.iterdir():
            if img_file.is_file():
                shutil.copy2(img_file, OUTPUT_DIR / "img" / img_file.name)

    logger.info(
        "Site built: %d articles from %d sources, output in %s",
        len(articles), source_count, OUTPUT_DIR,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("Starting AI Legal News build...")

    config = load_feed_config()
    feed_configs = config["feeds"]

    # Resolve keyword shorthands ("ai"/"legal") to full keyword lists
    for fc in feed_configs:
        fc["_resolved_keywords"] = _resolve_keywords(fc, config)

    logger.info("Loaded %d feed sources", len(feed_configs))

    all_articles = fetch_all_feeds(feed_configs)
    logger.info("Fetched %d total articles", len(all_articles))

    # Filter by age
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_ARTICLE_AGE_DAYS)
    all_articles = [a for a in all_articles if a["published"] >= cutoff]
    logger.info("%d articles after age filter", len(all_articles))

    # Deduplicate within current batch
    all_articles = deduplicate(all_articles)
    logger.info("%d articles after dedup", len(all_articles))

    # Update seen tracking
    seen = load_seen_articles()
    now_iso = datetime.now(timezone.utc).isoformat()
    for article in all_articles:
        if article["id"] not in seen:
            seen[article["id"]] = now_iso
    save_seen_articles(seen)

    # Build
    build_site(all_articles)
    logger.info("Build complete!")


if __name__ == "__main__":
    main()
