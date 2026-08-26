"""HTML → markdown extraction for the web fetch tool.

Converts fetched HTML into markdown suitable for a model to read. Non-content
elements are removed and ``data:`` URI images are dropped so large inline blobs
do not bloat the output. The page title is extracted separately from the ``<head>``.
"""

from __future__ import annotations

import logging

try:
    from bs4 import BeautifulSoup, Tag
    from markdownify import MarkdownConverter
except ImportError as e:
    raise ImportError(
        "web_fetch requires the 'web-fetch' extra (markdownify, beautifulsoup4). "
        "Install with: pip install 'strands-agents[web-fetch]'"
    ) from e

logger = logging.getLogger(__name__)

_DROPPED_ELEMENTS = [
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "iframe",
    "object",
    "embed",
    "video",
    "audio",
    "form",
    "input",
    "button",
    "select",
    "textarea",
]


def _tag_attribute(tag: Tag, name: str) -> str:
    """Return ``tag[name]`` as a single string ("" if absent), joining multi-valued attributes."""
    value = tag.get(name)
    if isinstance(value, list):
        return " ".join(value)
    return value or ""


def _sanitize_tree(soup: BeautifulSoup) -> None:
    """Remove non-content elements and unsafe/oversized links and images in place."""
    for element in soup(_DROPPED_ELEMENTS):
        element.decompose()
    for image in soup.find_all("img"):
        source = _tag_attribute(image, "src").lstrip().lower()
        # Image sources can be enormous blobs, so replace them with their alt text.
        if source.startswith("data:"):
            image.replace_with(_tag_attribute(image, "alt"))
        # javascript: sources are never useful to a model, so drop them outright.
        elif source.startswith("javascript:"):
            image.decompose()
    # Unwrap javascript: links to their text so the scheme never reaches output.
    for anchor in soup.find_all("a"):
        if _tag_attribute(anchor, "href").lstrip().lower().startswith("javascript:"):
            anchor.unwrap()


def html_to_markdown(html: str) -> tuple[str, str]:
    """Convert HTML to markdown suitable for a model to read.

    Returns:
        Stripped ``(title, markdown)``, or ``("", "")`` on a parser error.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else ""
        # Drop the head so its metadata does not leak into the body markdown.
        if soup.head is not None:
            soup.head.decompose()
        _sanitize_tree(soup)
        markdown = MarkdownConverter(heading_style="ATX", bullets="-").convert_soup(soup).strip()
    except Exception:
        logger.debug("html_to_markdown conversion raised; returning empty output", exc_info=True)
        return "", ""
    return title, markdown
