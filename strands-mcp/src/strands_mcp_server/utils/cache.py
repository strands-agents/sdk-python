import os
import threading
from typing import Dict

from ..config import doc_config
from . import doc_fetcher, indexer, text_processor

# Global state
_INDEX: indexer.IndexSearch | None = None
_URL_CACHE: Dict[str, doc_fetcher.Page | None] = {}  # url -> Page (None if not fetched yet)
_URL_TITLES: Dict[str, str] = {}  # url -> curated title from llms.txt
_LINKS_LOADED = False
_PREFETCH_STARTED = False

SNIPPET_HYDRATE_MAX = 5  # how many top results to hydrate with content

# Environment variable to enable background prefetch (default: OFF)
PREFETCH_ENV_VAR = "STRANDS_MCP_PREFETCH_ALL"


def _is_prefetch_enabled() -> bool:
    """Check if background prefetch is enabled via environment variable.

    Returns:
        True if STRANDS_MCP_PREFETCH_ALL is set to a truthy value (1, true, yes)
    """
    val = os.environ.get(PREFETCH_ENV_VAR, "").lower()
    return val in ("1", "true", "yes")


def _background_prefetch() -> None:
    """Background thread function to prefetch all pages.

    Iterates through all URLs in the cache and fetches content for any
    that haven't been fetched yet. Updates the index with hydrated content.
    Note: Search may return transiently stale results during hydration (eventual consistency).
    """
    global _URL_CACHE, _INDEX
    for url in list(_URL_CACHE.keys()):
        if _URL_CACHE.get(url) is None:
            try:
                ensure_page(url)
            except Exception:
                pass  # Silently skip failures in background prefetch


def _start_background_prefetch() -> None:
    """Start background prefetch thread if enabled and not already started.

    This function is idempotent - calling it multiple times has no effect
    after the first successful start.
    """
    global _PREFETCH_STARTED
    if _PREFETCH_STARTED or not _is_prefetch_enabled():
        return

    _PREFETCH_STARTED = True
    thread = threading.Thread(target=_background_prefetch, daemon=True)
    thread.start()


def load_links_only() -> None:
    """Parse llms.txt files and index curated titles without fetching content.

    This function initializes the search index with document titles and URLs from
    configured llms.txt files. Content is not fetched during initialization for
    faster startup times.

    If STRANDS_MCP_PREFETCH_ALL environment variable is set to a truthy value,
    background prefetch of all pages will be started after initialization.

    Side Effects:
        - Updates global _INDEX with document entries
        - Populates _URL_TITLES with curated titles
        - Sets placeholder entries in _URL_CACHE
        - Sets _LINKS_LOADED to True
        - May start background prefetch thread if enabled
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

    # Start background prefetch if enabled
    _start_background_prefetch()


def ensure_ready() -> None:
    """Ensure the search index is initialized and ready for use.

    Calls load_links_only() if the index hasn't been loaded yet.
    This is the main entry point for index initialization.
    """
    if not _LINKS_LOADED:
        load_links_only()


def ensure_page(url: str) -> doc_fetcher.Page | None:
    """Ensure a page is cached, fetching it if necessary.

    When a page is fetched, also updates the search index with the hydrated content
    so that body-only terms become searchable. This operation is idempotent -
    re-fetching an already-cached page has no effect.

    Thread Safety:
        If indexing fails after fetch succeeds, the page is NOT cached, allowing
        retry on subsequent calls. This ensures body terms eventually become
        searchable even after transient indexing failures.

    Args:
        url: The URL of the page to ensure is cached

    Returns:
        The cached or newly fetched Page object, or None if fetch failed

    """
    page = _URL_CACHE.get(url)
    if page is not None:
        return page
    try:
        raw = doc_fetcher.fetch_and_clean(url)
        display_title = text_processor.format_display_title(url, raw.title, _URL_TITLES)
        page = doc_fetcher.Page(url=url, title=display_title, content=raw.content)

        # Update the search index with the hydrated content BEFORE caching
        # If this fails, we leave the page un-cached so retry is possible
        if _INDEX is not None:
            _INDEX.update_content(url, raw.content)

        # Only cache after indexing succeeds
        _URL_CACHE[url] = page

        return page
    except Exception:
        return None


def get_index() -> indexer.IndexSearch | None:
    """Get the current search index instance.

    Returns:
        The initialized IndexSearch instance, or None if not yet loaded
    """
    return _INDEX


def get_url_cache() -> Dict[str, doc_fetcher.Page | None]:
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
