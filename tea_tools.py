"""Tea catalog search over local Shopify-style JSON (tea_data.json)."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from agno.tools import tool

TEA_DATA_PATH = "tea_data.json"
_MAX_RESULTS = 25

_STOPWORDS = frozenset(
    """
    a an the and or for with from that this these those are was were been being
    have has had do does did will would could should may might must can need
    your you we they i me my our us them their what which who how when where
    why show find list tell give get make any all some most more less few
    very just only also into out up down about over under please want looking
    teas sales sale sold item items product products shop store harney
    sons fine dollars dollar buck bucks each per case bottle tin bag sachet
    ounce oz grams free shipping not available
    under below above between around than more less cheapest pricey budget
    one two three four five six seven eight nine ten eleven twelve twenty
    thirty forty fifty sixty hundred thousand
    """.split()
)


def _tea_search_cli_banner() -> None:
    print(
        "\n  *** Barney's ledger: search_tea_inventory invoked ***\n",
        flush=True,
    )


def _tea_search_cli_done() -> None:
    print("  *** search_tea_inventory finished ***\n", flush=True)


@lru_cache(maxsize=1)
def _load_products() -> tuple[dict[str, Any], ...]:
    path = Path(__file__).resolve().parent / TEA_DATA_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Tea catalog not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    products = data.get("products")
    if not isinstance(products, list):
        raise ValueError("tea_data.json: missing 'products' array")
    return tuple(products)


def _strip_html(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html or "")


def _search_blob(product: dict[str, Any]) -> str:
    parts = [
        str(product.get("title") or ""),
        str(product.get("handle") or ""),
        str(product.get("vendor") or ""),
        str(product.get("product_type") or ""),
        _strip_html(str(product.get("body_html") or "")),
    ]
    tags = product.get("tags")
    if isinstance(tags, list):
        parts.extend(str(t) for t in tags)
    return " ".join(parts).lower()


def _variant_price(variant: dict[str, Any]) -> float:
    try:
        return float(variant.get("price") or 0)
    except (TypeError, ValueError):
        return 0.0


def _variant_on_sale(variant: dict[str, Any]) -> bool:
    cap = variant.get("compare_at_price")
    if cap is None or cap == "":
        return False
    try:
        return float(cap) > _variant_price(variant)
    except (TypeError, ValueError):
        return False


def _infer_caffeine(blob: str) -> str:
    """Coarse bucket from catalog text (no structured caffeine field in JSON)."""
    b = blob.lower()
    if re.search(
        r"\bdecaffeinated\b|\bdecaf\b|caffeine-free|caffeine free|"
        r"no caffeine|none of the caffeine|zero caffeine",
        b,
    ):
        return "none"
    if "very low in caffeine" in b or "low in caffeine" in b or "lightly caffeinated" in b:
        return "low"
    if re.search(r"\b50\s*mg\b.*caffeine|caffeine:\s*50\s*mg", b):
        return "high"
    if re.search(
        r"\b(herbal|tisane|rooibos|chamomile)\b.*caffeine-free|caffeine-free.*\b(herbal|rooibos)",
        b,
    ):
        return "none"
    if re.search(
        r"naturally caffeinated|caffeinated\.|caffeinated for|"
        r"\b(black tea|green tea|oolong|matcha|sencha|pu-erh|puerh|white tea)\b",
        b,
    ):
        return "medium"
    if "some caffeine" in b:
        return "medium"
    if re.search(r"\b(black tea|green tea|oolong|matcha|white tea|sencha)\b", b):
        return "medium"
    if re.search(r"caffeine-free|decaf\b|decaffeinated", b):
        return "none"
    if re.search(r"\b(herbal|tisane|rooibos|chamomile)\b", b):
        return "none"
    if re.search(
        r"\b(english breakfast|earl grey|earl gray|chai|assam|darjeeling|keemun|"
        r"matcha|jasmine|oolong)\b",
        b,
    ):
        return "medium"
    if re.search(r"\btea\b", b):
        return "medium"
    return "unknown"


def _normalize_caffeine_filter(raw: Optional[str]) -> Optional[str]:
    if raw is None or not str(raw).strip():
        return None
    x = str(raw).strip().lower()
    aliases = {
        "zero": "none",
        "free": "none",
        "decaf": "none",
        "caffeine-free": "none",
        "caffeine free": "none",
        "no caffeine": "none",
    }
    return aliases.get(x, x)


def _caffeine_matches(blob: str, wanted_raw: Optional[str]) -> bool:
    wanted = _normalize_caffeine_filter(wanted_raw)
    if wanted is None or wanted in ("any", "unknown"):
        return True
    inferred = _infer_caffeine(blob)
    if wanted == "none":
        return inferred == "none"
    if wanted == "low":
        return inferred in ("none", "low")
    if wanted == "medium":
        return inferred in ("medium", "high", "low")
    if wanted == "high":
        return inferred in ("high", "medium")
    return True


def _intent_tokens(intent_summary: str) -> list[str]:
    words = re.findall(r"[a-zA-Z]{3,}", intent_summary.lower())
    return [w for w in words if w not in _STOPWORDS]


def _intent_matches_blob(blob: str, intent_summary: str) -> bool:
    tokens = _intent_tokens(intent_summary)
    if not tokens:
        return True
    return all(t in blob for t in tokens)


def _ingredients_match(blob: str, ingredients: Optional[list[str]]) -> bool:
    if not ingredients:
        return True
    for ing in ingredients:
        if not ing or not str(ing).strip():
            continue
        if str(ing).strip().lower() not in blob:
            return False
    return True


def _variant_passes_filters(
    variant: dict[str, Any],
    on_sale_only: Optional[bool],
    price_min: Optional[float],
    price_max: Optional[float],
) -> bool:
    price = _variant_price(variant)
    if price_min is not None and price < price_min:
        return False
    if price_max is not None and price > price_max:
        return False
    if on_sale_only is True and not _variant_on_sale(variant):
        return False
    if on_sale_only is False and _variant_on_sale(variant):
        return False
    return True


def _format_variant(v: dict[str, Any]) -> dict[str, Any]:
    price = _variant_price(v)
    on_sale = _variant_on_sale(v)
    cap = v.get("compare_at_price")
    out: dict[str, Any] = {
        "id": v.get("id"),
        "variant_title": v.get("title"),
        "sku": v.get("sku"),
        "price": price,
        "currency": "USD",
        "compare_at_price": float(cap) if cap not in (None, "") else None,
        "on_sale": on_sale,
        "available": v.get("available"),
        "review_count": None,
    }
    if on_sale and cap not in (None, ""):
        try:
            out["discount_pct"] = round(100 * (1 - price / float(cap)), 1)
        except (TypeError, ValueError, ZeroDivisionError):
            out["discount_pct"] = None
    return out


def _search_catalog(
    intent_summary: str,
    on_sale_only: Optional[bool],
    price_min: Optional[float],
    price_max: Optional[float],
    min_review_count: Optional[int],
    ingredients_include: Optional[list[str]],
    caffeine_level: Optional[str],
) -> dict[str, Any]:
    notes: list[str] = []
    if min_review_count is not None and min_review_count > 0:
        notes.append(
            "min_review_count is not supported: tea_data.json has no review-count fields; "
            "that filter was ignored."
        )

    products = _load_products()
    rows: list[dict[str, Any]] = []

    for product in products:
        blob = _search_blob(product)
        if not _caffeine_matches(blob, caffeine_level):
            continue
        if not _ingredients_match(blob, ingredients_include or []):
            continue
        if not _intent_matches_blob(blob, intent_summary):
            continue

        variants_out: list[dict[str, Any]] = []
        for v in product.get("variants") or []:
            if not isinstance(v, dict):
                continue
            if not _variant_passes_filters(v, on_sale_only, price_min, price_max):
                continue
            variants_out.append(_format_variant(v))

        if not variants_out:
            continue

        rows.append(
            {
                "id": product.get("id"),
                "title": product.get("title"),
                "handle": product.get("handle"),
                "vendor": product.get("vendor"),
                "tags": product.get("tags"),
                "caffeine_inferred": _infer_caffeine(blob),
                "variants": variants_out,
            }
        )

    truncated = len(rows) > _MAX_RESULTS
    rows = rows[:_MAX_RESULTS]
    if truncated:
        notes.append(f"Results truncated to {_MAX_RESULTS} products; refine filters to narrow.")

    return {
        "intent_summary": intent_summary,
        "filters_applied": {
            "on_sale_only": on_sale_only,
            "price_min": price_min,
            "price_max": price_max,
            "min_review_count": min_review_count,
            "ingredients_include": ingredients_include,
            "caffeine_level": caffeine_level,
        },
        "match_count": len(rows),
        "notes": notes,
        "products": rows,
    }


@tool(
    name="search_tea_inventory",
    description=(
        "Search the shop tea catalog (tea_data.json). Filter by sale status (variant compare_at vs price), "
        "price range, ingredient substrings in title/body/tags, and inferred caffeine level "
        "(none/low/medium/high from text). Combine filters as needed. Review counts are not in this export."
    ),
    add_instructions=False,
    pre_hook=_tea_search_cli_banner,
    post_hook=_tea_search_cli_done,
)
def search_tea_inventory(
    intent_summary: str,
    on_sale_only: Optional[bool] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    min_review_count: Optional[int] = None,
    ingredients_include: Optional[list[str]] = None,
    caffeine_level: Optional[str] = None,
) -> str:
    """Return JSON string of matching products and variant metrics for the model."""
    result = _search_catalog(
        intent_summary=intent_summary,
        on_sale_only=on_sale_only,
        price_min=price_min,
        price_max=price_max,
        min_review_count=min_review_count,
        ingredients_include=ingredients_include,
        caffeine_level=caffeine_level,
    )
    return json.dumps(result, indent=2, ensure_ascii=False)
