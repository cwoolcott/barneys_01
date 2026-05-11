"""Harney.com (Shopify) cart deep-links from local catalog ids."""

from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from typing import Optional

from agno.tools import tool

from env_util import load_app_env

HARNEY_STORE_BASE = os.getenv("HARNEY_STORE_BASE", "https://www.harney.com").rstrip("/")
_MAX_QTY = 20


def _cart_cli_banner() -> None:
    print("\n  *** harney_add_to_cart (cart link) ***\n", flush=True)


def _cart_cli_done() -> None:
    print("  *** harney_add_to_cart finished ***\n", flush=True)


def _load_products_tuple():
    from tea_tools import _load_products

    return _load_products()


def _find_variant(
    *,
    variant_id: Optional[int],
    product_handle: Optional[str],
    variant_keyword: Optional[str],
) -> tuple[Optional[dict], Optional[dict], str | None]:
    """Return (product, variant, error)."""
    products = _load_products_tuple()

    if variant_id is not None:
        vid = int(variant_id)
        for p in products:
            for v in p.get("variants") or []:
                if not isinstance(v, dict):
                    continue
                try:
                    if int(v.get("id") or 0) == vid:
                        return p, v, None
                except (TypeError, ValueError):
                    continue
        return None, None, f"No variant with id={vid} in local tea_data.json."

    handle = (product_handle or "").strip().lower()
    if not handle:
        return None, None, "Provide `variant_id` (from catalog JSON) or `product_handle` (Shopify handle)."

    kw = (variant_keyword or "").strip().lower()
    for p in products:
        ph = str(p.get("handle") or "").lower()
        if ph != handle:
            continue
        variants = [v for v in (p.get("variants") or []) if isinstance(v, dict)]
        if not variants:
            return p, None, f"Product handle {handle!r} has no variants in catalog."
        if kw:
            for v in variants:
                title = str(v.get("title") or "").lower()
                sku = str(v.get("sku") or "").lower()
                if kw in title or kw in sku:
                    return p, v, None
            return p, None, f"No variant on {handle!r} matches keyword {variant_keyword!r}."
        return p, variants[0], None

    return None, None, f"No product with handle={handle!r} in local tea_data.json."


def _shopify_cart_url(variant_id: int, quantity: int) -> str:
    return f"{HARNEY_STORE_BASE}/cart/{int(variant_id)}:{int(quantity)}"


def _product_url(handle: str) -> str:
    return f"{HARNEY_STORE_BASE}/products/{handle}"


def _open_cart_url(cart_url: str) -> tuple[bool, str | None]:
    """Open add-to-cart URL in the default browser (macOS `open` is most reliable)."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", cart_url], check=False, timeout=30)
            return True, None
        if sys.platform == "win32":
            os.startfile(cart_url)  # noqa: S606
            return True, None
        if sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", cart_url], check=False, timeout=30)
            return True, None
        if webbrowser.open(cart_url):
            return True, None
        return False, "webbrowser.open returned False"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


@tool(
    name="harney_add_to_cart",
    description=(
        "Build a Harney.com (Shopify) add-to-cart URL and **open it in the default browser by default** "
        "(same machine), which triggers Shopify to add the line item. Use `variant_id` from "
        "`search_tea_inventory` (`variants[].id`), or `product_handle` plus optional `variant_keyword`. "
        "Set `open_in_browser` false to only return links without opening a window. "
        "Set env `HARNEY_AUTO_OPEN_CART=0` to disable auto-open globally (e.g. SSH)."
    ),
    add_instructions=False,
    pre_hook=_cart_cli_banner,
    post_hook=_cart_cli_done,
)
def harney_add_to_cart(
    variant_id: Optional[int] = None,
    product_handle: Optional[str] = None,
    variant_keyword: Optional[str] = None,
    quantity: int = 1,
    open_in_browser: bool = True,
) -> str:
    load_app_env()
    if quantity < 1 or quantity > _MAX_QTY:
        return f"harney_add_to_cart: quantity must be 1–{_MAX_QTY}."

    product, variant, err = _find_variant(
        variant_id=variant_id,
        product_handle=product_handle,
        variant_keyword=variant_keyword,
    )
    if err or not variant:
        return f"harney_add_to_cart: {err or 'variant not found.'}"

    vid = variant.get("id")
    if vid is None:
        return "harney_add_to_cart: variant has no id."
    try:
        vid_int = int(vid)
    except (TypeError, ValueError):
        return "harney_add_to_cart: invalid variant id."

    handle = str(product.get("handle") or "")
    title = str(product.get("title") or "")
    vtitle = str(variant.get("title") or "")
    sku = str(variant.get("sku") or "")
    cart_url = _shopify_cart_url(vid_int, quantity)
    product_url = _product_url(handle) if handle else ""

    env_auto = os.getenv("HARNEY_AUTO_OPEN_CART", "1").strip().lower()
    env_allows_open = env_auto not in ("0", "false", "no")
    do_open = bool(open_in_browser) and env_allows_open

    opened = ""
    if do_open:
        ok, oerr = _open_cart_url(cart_url)
        if ok:
            opened = "\n\n_Opened the add-to-cart link in your default browser (Shopify should show the cart)._"
        else:
            opened = f"\n\n_Auto-open failed ({oerr or 'unknown'}). Use the link below manually._"

    lines = [
        "## Harney.com — add to cart",
        "",
        f"**{title}** — _{vtitle}_",
        "",
        f"- **Variant ID:** `{vid_int}`",
        f"- **Quantity:** {quantity}",
    ]
    if sku:
        lines.append(f"- **SKU:** `{sku}`")
    lines.extend(
        [
            "",
            "### One-click cart (Shopify)",
            "",
            "This URL adds the line item when opened in a browser (auto-open is on by default in this tool):",
            "",
            f"[**Add to cart on harney.com**]({cart_url})",
            "",
            f"`{cart_url}`",
        ]
    )
    if product_url:
        lines.extend(["", "### Product page", "", f"[View product]({product_url})"])
    lines.append("")
    lines.append(
        "_If the cart looks empty, allow cookies for harney.com or try again in a private window._"
    )
    if opened:
        lines.append(opened)
    return "\n".join(lines)
