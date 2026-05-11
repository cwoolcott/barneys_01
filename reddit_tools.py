"""Reddit sentiment: Reddit search.json + Firecrawl scrape + OpenAI summary."""

from __future__ import annotations

import json
import os
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import requests
from openai import OpenAI

from agno.tools import tool

from env_util import load_app_env

REDDIT_SEARCH_JSON = "https://www.reddit.com/search.json"
FIRECRAWL_SCRAPE_URL = os.getenv("FIRECRAWL_SCRAPE_URL", "https://api.firecrawl.dev/v2/scrape")
REDDIT_USER_AGENT = os.getenv(
    "REDDIT_USER_AGENT",
    "BarneyTeaBot/1.0 (tea shop assistant; sentiment research)",
)
_MAX_THREADS = 5
_MAX_MARKDOWN_CHARS = 8000
_MAX_CORPUS_CHARS = 45000


def _harney_scoped_reddit_query(search_query: str) -> str:
    """Build a Reddit `q` string so results target Harney & Sons / Harney teas only."""
    q = " ".join(search_query.split())
    if not q:
        return '"Harney & Sons" tea'
    low = q.lower()
    if "harney" not in low and "harneys" not in low:
        q = f'{q} ("Harney & Sons" OR Harney OR Harneys)'
    return q


def _post_references_harney(data: dict) -> bool:
    """True if the Reddit post metadata clearly references Harney (brand / shop teas)."""
    parts = [
        str(data.get("title") or ""),
        str(data.get("selftext") or ""),
        str(data.get("url") or ""),
        str(data.get("domain") or ""),
        str(data.get("link_flair_text") or ""),
    ]
    blob = " ".join(parts).lower()
    if "harney" in blob or "harneys" in blob:
        return True
    if "h&s" in blob and "tea" in blob:
        return True
    return False


def _reddit_cli_banner() -> None:
    print("\n  *** redditsearch invoked (Reddit JSON + Firecrawl + summary) ***\n", flush=True)


def _reddit_cli_done() -> None:
    print("  *** redditsearch finished ***\n", flush=True)


def _reddit_search_thread_urls(query: str, limit: int = _MAX_THREADS) -> tuple[list[str], str | None]:
    """Return canonical thread URLs from Reddit search.json."""
    params = {
        "q": query,
        "limit": str(max(50, limit * 8)),
        "sort": "relevance",
        "raw_json": "1",
    }
    url = f"{REDDIT_SEARCH_JSON}?{urllib.parse.urlencode(params)}"
    try:
        r = requests.get(
            url,
            headers={"User-Agent": REDDIT_USER_AGENT},
            timeout=30,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        return [], f"Reddit search failed: {e}"

    try:
        payload = r.json()
    except json.JSONDecodeError as e:
        return [], f"Reddit returned non-JSON: {e}"

    children = (payload.get("data") or {}).get("children") or []
    urls: list[str] = []
    seen: set[str] = set()
    for child in children:
        if not isinstance(child, dict) or child.get("kind") != "t3":
            continue
        data = child.get("data") or {}
        if data.get("promoted") or data.get("is_promoted"):
            continue
        if not _post_references_harney(data):
            continue
        permalink = data.get("permalink")
        if not permalink or not isinstance(permalink, str):
            continue
        if not permalink.startswith("/"):
            permalink = "/" + permalink
        full = "https://www.reddit.com" + permalink.split("?")[0]
        pid = str(data.get("id") or full)
        if pid in seen:
            continue
        seen.add(pid)
        urls.append(full)
        if len(urls) >= limit:
            break

    if not urls:
        return [], "No Reddit thread URLs found for that query (try broader keywords)."
    return urls, None


def _thread_url_to_json_url(thread_url: str) -> str | None:
    base = thread_url.split("?")[0].rstrip("/")
    if "reddit.com" not in base.lower() or "/comments/" not in base:
        return None
    return f"{base}.json?raw_json=1&limit=200"


def _flatten_comment_tree(children: list[Any], lines: list[str], depth: int = 0, max_depth: int = 6) -> None:
    if depth > max_depth:
        return
    indent = "  " * min(depth, 4)
    for ch in children or []:
        if not isinstance(ch, dict):
            continue
        if ch.get("kind") != "t1":
            continue
        data = ch.get("data") or {}
        body = (data.get("body") or "").strip()
        if not body or body in ("[deleted]", "[removed]"):
            continue
        author = data.get("author") or "?"
        lines.append(f"{indent}- **u/{author}:** {body}\n")
        replies = data.get("replies")
        if isinstance(replies, dict):
            nested = (replies.get("data") or {}).get("children")
            if nested:
                _flatten_comment_tree(nested, lines, depth + 1, max_depth)


def _reddit_thread_markdown_via_json(thread_url: str) -> tuple[str, str | None]:
    """Fetch post + comments via Reddit’s public .json API (works when HTML scrapers get 403)."""
    json_url = _thread_url_to_json_url(thread_url)
    if not json_url:
        return "", "Not a Reddit thread URL; cannot use .json fallback."

    try:
        r = requests.get(
            json_url,
            headers={"User-Agent": REDDIT_USER_AGENT},
            timeout=45,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        return "", f"Reddit .json fetch failed: {e}"

    try:
        payload = r.json()
    except json.JSONDecodeError as e:
        return "", f"Reddit .json parse error: {e}"

    if not isinstance(payload, list) or len(payload) < 1:
        return "", "Unexpected Reddit .json shape."

    lines: list[str] = []
    try:
        post_wrap = payload[0]["data"]["children"][0]["data"]
        title = post_wrap.get("title") or ""
        sub = post_wrap.get("subreddit") or ""
        selftext = (post_wrap.get("selftext") or "").strip()
        lines.append(f"## {title}\n")
        lines.append(f"*r/{sub}*\n\n")
        if selftext:
            lines.append(selftext + "\n\n")
        lines.append("### Comments\n\n")
        if len(payload) > 1:
            children = (payload[1].get("data") or {}).get("children") or []
            _flatten_comment_tree(children, lines)
    except (KeyError, IndexError, TypeError) as e:
        return "", f"Reddit .json extraction error: {e}"

    md = "".join(lines).strip()
    if not md:
        return "", "Reddit .json returned no extractable text."
    if len(md) > _MAX_MARKDOWN_CHARS:
        md = md[:_MAX_MARKDOWN_CHARS] + "\n\n[…truncated…]"
    return md, None


def _firecrawl_scrape_markdown_only(api_key: str, page_url: str) -> tuple[str, str | None]:
    """Scrape one URL via Firecrawl only; returns (markdown or empty, error or None)."""
    try:
        r = requests.post(
            FIRECRAWL_SCRAPE_URL,
            json={"url": page_url, "formats": ["markdown"]},
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=90,
        )
    except requests.RequestException as e:
        return "", f"Firecrawl request error for {page_url}: {e}"

    try:
        body = r.json()
    except json.JSONDecodeError:
        return "", f"Firecrawl non-JSON ({r.status_code}) for {page_url}: {r.text[:400]}"

    if r.status_code == 401:
        return "", "Firecrawl returned 401 — check FIRECRAWL_API_KEY in .env."
    if r.status_code >= 400 or not body.get("success"):
        err = body.get("error") or body.get("message") or r.text[:500]
        low = str(err).lower()
        if r.status_code == 403 or "forbidden" in low or "blocked" in low:
            return "", f"Firecrawl forbidden/blocked ({r.status_code}) for {page_url}: {err}"
        return "", f"Firecrawl error ({r.status_code}) for {page_url}: {err}"

    md = (body.get("data") or {}).get("markdown") or ""
    if len(md) > _MAX_MARKDOWN_CHARS:
        md = md[:_MAX_MARKDOWN_CHARS] + "\n\n[…truncated…]"
    return md, None


def _fetch_thread_markdown(api_key: str, page_url: str) -> tuple[str, str | None]:
    """
    Prefer Firecrawl-rendered markdown; if Reddit returns 403 to scrapers or Firecrawl fails,
    fall back to Reddit’s official .json thread export (same User-Agent as search).
    """
    fc_md, fc_err = _firecrawl_scrape_markdown_only(api_key, page_url)
    if fc_md and len(fc_md.strip()) >= 120:
        return fc_md, None

    r_md, r_err = _reddit_thread_markdown_via_json(page_url)
    if r_md and len(r_md.strip()) >= 40:
        note = (
            f"> _Thread text loaded via Reddit’s `.json` API (Firecrawl unusable here: "
            f"{fc_err or 'empty markdown'})._ \n\n"
        )
        return note + r_md, None

    return "", fc_err or r_err or "No thread content from Firecrawl or Reddit .json."


def _scrape_threads_parallel(api_key: str, urls: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_map = {pool.submit(_fetch_thread_markdown, api_key, u): u for u in urls}
        for fut in as_completed(fut_map):
            u = fut_map[fut]
            try:
                md, err = fut.result()
            except Exception as e:  # noqa: BLE001
                md, err = "", str(e)
            out.append({"url": u, "markdown": md, "error": err})
    out.sort(key=lambda x: urls.index(x["url"]))
    return out


def _summarize_reddit_threads(
    *,
    search_query: str,
    reddit_q_used: str,
    context: str | None,
    bundles: list[dict[str, Any]],
) -> str:
    load_app_env()
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return _fallback_report(search_query, reddit_q_used, bundles)

    if not any((b.get("markdown") or "").strip() for b in bundles):
        lines = [
            "## Reddit sentiment — fetch failed",
            "",
            f"**Shopper topic:** {search_query}",
            "",
            f"**Reddit `q` used:** {reddit_q_used}",
            "",
            "No thread content was retrieved (all Firecrawl scrapes empty or errored). Per-thread status:",
        ]
        for b in bundles:
            lines.append(f"- [{b['url']}]({b['url']}) — {b.get('error') or 'empty body'}")
        lines.append("")
        lines.append("Fix `FIRECRAWL_API_KEY` / billing or try again later; do not invent Reddit quotes.")
        return "\n".join(lines)

    parts: list[str] = []
    for b in bundles:
        parts.append(f"--- THREAD: {b['url']}\n{b.get('markdown') or '(no content)'}\n")
    corpus = "\n".join(parts)
    if len(corpus) > _MAX_CORPUS_CHARS:
        corpus = corpus[:_MAX_CORPUS_CHARS] + "\n\n[…corpus truncated for model context…]"

    sys_msg = (
        "You read Reddit thread markdown (posts and comments). "
        "Threads are all about Harney & Sons / Harney teas only — stay on that brand. "
        "Reply with a single JSON object (no markdown fences) with keys: "
        "`overall_sentiment` (short phrase), "
        "`summary` (2–5 sentences on what Reddit users seem to think), "
        "`quotes` (array of up to 5 objects, each with `text` and `thread_url`). "
        "Each quote.text must be a short verbatim or near-verbatim snippet from the THREAD sections; "
        "thread_url must exactly match one of the THREAD URLs given. "
        "If the source is thin, return fewer quotes."
    )
    user_msg = f"Shopper topic: {search_query!r}\nReddit search used: {reddit_q_used!r}\n"
    if context:
        user_msg += f"Extra context from shopper: {context!r}\n"
    user_msg += "\nSource threads and content:\n" + corpus

    client = OpenAI(api_key=openai_key)
    try:
        comp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = (comp.choices[0].message.content or "").strip()
        data = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        return _fallback_report(search_query, reddit_q_used, bundles) + f"\n\n(OpenAI summarization failed: {e})"

    lines = [
        "## Reddit sentiment (tool) — Harney’s teas only",
        "",
        f"**Shopper topic:** {search_query}",
        "",
        f"**Reddit `q` used:** {reddit_q_used}",
        "",
        f"**Overall sentiment:** {data.get('overall_sentiment', 'n/a')}",
        "",
        f"**Summary:** {data.get('summary', 'n/a')}",
        "",
        "### Quotes",
    ]
    quotes = data.get("quotes")
    if isinstance(quotes, list):
        allowed = {b["url"] for b in bundles}
        for i, q in enumerate(quotes[:5], 1):
            if not isinstance(q, dict):
                continue
            text = (q.get("text") or "").strip()
            link = (q.get("thread_url") or "").strip()
            if link not in allowed:
                for u in allowed:
                    if link in u or u in link:
                        link = u
                        break
            if not text:
                continue
            lines.append(f"{i}. “{text}”")
            lines.append(f"   — [{link}]({link})")
    else:
        lines.append("_No quotes returned._")

    lines.extend(["", "### Threads scraped"])
    for b in bundles:
        status = "ok" if b.get("markdown") and not b.get("error") else (b.get("error") or "empty")
        lines.append(f"- [{b['url']}]({b['url']}) — {status}")

    return "\n".join(lines)


def _fallback_report(search_query: str, reddit_q_used: str, bundles: list[dict[str, Any]]) -> str:
    lines = [
        "## Reddit sentiment (fallback — set OPENAI_API_KEY for AI summary)",
        "",
        f"**Shopper topic:** {search_query}",
        "",
        f"**Reddit `q` used:** {reddit_q_used}",
        "",
        "### Threads scraped",
    ]
    for b in bundles:
        err = b.get("error")
        preview = (b.get("markdown") or "")[:600].replace("\n", " ")
        if err:
            lines.append(f"- {b['url']} — error: {err}")
        else:
            lines.append(f"- {b['url']} — preview: {preview}…")
    return "\n".join(lines)


@tool(
    name="redditsearch",
    description=(
        "Harney-focused Reddit read: Reddit search.json for threads, then thread text via Firecrawl when "
        "available, otherwise Reddit’s official `.json` thread export (Reddit/Firecrawl often block HTML "
        "scraping). Summarizes sentiment with quotes and links. Pass tea name or topic; Harney scope is added "
        "automatically."
    ),
    add_instructions=False,
    pre_hook=_reddit_cli_banner,
    post_hook=_reddit_cli_done,
)
def redditsearch(
    search_query: str,
    context: Optional[str] = None,
) -> str:
    """
    Args:
        search_query: Tea topic for Reddit (e.g. \"Moroccan Mint\", \"Sencha\"); Harney scope is applied automatically.
        context: Optional shopper nuance (comparison, variant, concern).
    """
    load_app_env()
    if not search_query or not str(search_query).strip():
        return "redditsearch: empty search_query."

    fc_key = os.getenv("FIRECRAWL_API_KEY")
    if not fc_key:
        return "redditsearch: missing FIRECRAWL_API_KEY in environment (.env)."

    raw_q = str(search_query).strip()
    reddit_q = _harney_scoped_reddit_query(raw_q)
    urls, err = _reddit_search_thread_urls(reddit_q)
    if err:
        return f"redditsearch: {err}"

    bundles = _scrape_threads_parallel(fc_key.strip(), urls)
    return _summarize_reddit_threads(
        search_query=raw_q,
        reddit_q_used=reddit_q,
        context=context,
        bundles=bundles,
    )
