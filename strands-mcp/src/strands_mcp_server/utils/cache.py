import logging
import time
from typing import Any, Dict

from ..config import doc_config
from . import doc_fetcher, indexer, text_processor

logger = logging.getLogger(__name__)

# TTL for negatively cached (failed) fetches — transient failures are retried
# after this duration
_FAILED_TTL_SECONDS = 300  # 5 minutes


class _FailedEntry:
    """Sentinel for a negatively cached fetch failure with TTL expiry.

    A page whose fetch failed is stored as a _FailedEntry instead of
    being re-fetched immediately. Once TTL expires, the entry is treated
    as stale and the next ensure_page call will retry the fetch.
    """

    __slots__ = ("_timestamp",)

    def __init__(self) -> None:
        self._timestamp = time.monotonic()

    @property
    def expired(self) -> bool:
        return time.monotonic() - self._timestamp >= _FAILED_TTL_SECONDS


# Global state
_INDEX: indexer.IndexSearch | None = None
_URL_CACHE: Dict[str, Any] = {}  # url -> Page, None (not fetched), or _FailedEntry (failed)
_URL_TITLES: Dict[str, str] = {}  # url -> curated title from llms.txt
_LINKS_LOADED = False

SNIPPET_HYDRATE_MAX = 5  # how many top results to hydrate with content


def load_links_only() -> None:
    """Parse llms.txt files and index curated titles without fetching content.

    This function initializes the search index with document titles and URLs from
    configured llms.txt files. Content is not fetched during initialization for
    faster startup times.

    Side Effects:
        - Updates global _INDEX with document entries
        - Populates _URL_TITLES with curated titles
        - Sets placeholder entries in _URL_CACHE
        - Sets _LINKS_LOADED to True
    """
    global _INDEX, _LINKS_LOADED, _URL_TITLES, _URL_CACHE
    if _INDEX is None:
        _INDEX = indexer.IndexSearch()

    for src in doc_config.llm_texts_url:
        for title, url in doc_fetcher.parse_llms_txt(src):
            # Record curated display title and placeholder cache
            _URL_TITLES[url] = title
            _URL_CACHE.setdefault(url, None)

            # For curated titles from llms.txt, we already have the title
            display_title = text_processor.normalize(title)
            index_title = text_processor.index_title_variants(display_title, url)

            # Index now with clean display title + hidden index variants; empty content for now
            _INDEX.add(indexer.Doc(uri=url, display_title=display_title, content="", index_title=index_title))

    _LINKS_LOADED = True


def ensure_ready() -> None:
    """Ensure the search index is initialized and ready for use.

    Calls load_links_only() if the index hasn't been loaded yet.
    This is the main entry point for index initialization.
    """
    if not _LINKS_LOADED:
        load_links_only()


def ensure_page(url: str) -> doc_fetcher.Page | None:
    """Ensure a page is cached, fetching it if necessary.

    Args:
        url: The URL of the page to ensure is cached

    Returns:
        The cached or newly fetched Page object, or None if fetch failed

    """
    cached = _URL_CACHE.get(url)
    # Short-circuit on known transient failures — don't re-fetch
    if isinstance(cached, _FailedEntry):
        if cached.expired:
            # TTL expired — clear and fall through to re-fetch
            _URL_CACHE[url] = None
        else:
            return None
    else:
        if cached is not None:
            return cached
    try:
        raw = doc_fetcher.fetch_and_clean(url)
        display_title = text_processor.format_display_title(url, raw.title, _URL_TITLES)
        page = doc_fetcher.Page(url=url, title=display_title, content=raw.content)
        _URL_CACHE[url] = page
        return page
    except Exception:
        logger.exception("Failed to fetch page: %s", url)
        _URL_CACHE[url] = _FailedEntry()  # negatively cache the failure
        return None


def get_index() -> indexer.IndexSearch | None:
    """Get the current search index instance.

    Returns:
        The initialized IndexSearch instance, or None if not yet loaded
    """
    return _INDEX


def get_url_cache() -> Dict[str, Any]:
    """Get the URL cache dictionary.

    Returns:
        Dictionary mapping URLs to cached Page objects (or None if not fetched)
    """
    return _URL_CACHE


def get_url_titles() -> Dict[str, str]:
    """Get the curated URL titles mapping.

    Returns:
        Dictionary mapping URLs to their curated display titles from llms.txt
    """
    return _URL_TITLES
