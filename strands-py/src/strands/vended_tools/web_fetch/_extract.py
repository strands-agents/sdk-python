"""HTML → markdown extraction for the web fetch tool.

The goal is not a perfect readability heuristic — it is producing output that
is safe and useful for a model to read. We strip script, style, and other
non-content elements entirely, drop ``data:`` URI images (which can be huge
blobs), and preserve headings, links, lists, blockquotes, code, and paragraph
text as GitHub-flavored markdown.

Uses only the standard library (``html.parser``) so this tool adds no runtime
dependency and no third-party surface to audit.
"""

from __future__ import annotations

import logging
from html import unescape
from html.parser import HTMLParser

logger = logging.getLogger(__name__)

# Elements whose content is discarded entirely, not just unwrapped.
_DROPPED_ELEMENTS = frozenset(
    {
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
        # ``head`` is intentionally NOT dropped -- it contains ``title``, which
        # we want to extract. Its problematic children (``script``, ``style``)
        # are already in this set.
    }
)

# HTML void elements never fire a matching end tag, so they must not affect
# the drop-depth counter -- otherwise a void child inside a dropped element
# would leak the drop state forever (e.g. <form><input></form> would keep the
# extractor in "dropping" mode after </form>).
_VOID_ELEMENTS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
)

_HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

# Sentinel markers used during extraction to record where a blockquote begins
# and ends. A post-processing pass prefixes every line between the markers with
# ``> `` so nested block elements (e.g. ``<blockquote><p>...</p></blockquote>``)
# stay inside the blockquote in the emitted markdown.
_BQ_OPEN = "\x00BQ_OPEN\x00"
_BQ_CLOSE = "\x00BQ_CLOSE\x00"


class _MarkdownExtractor(HTMLParser):
    """Streaming HTML→markdown extractor.

    Not a general HTML-to-markdown converter — good enough for feeding a model.
    """

    def __init__(self) -> None:
        # convert_charrefs=True → &amp; etc. are decoded before handle_data is called.
        super().__init__(convert_charrefs=True)
        self._out: list[str] = []
        # Stack of currently-open block-level containers we care about. Used to
        # emit list markers with the right indent and to close blocks cleanly.
        self._drop_depth = 0  # >0 → inside script/style/etc., discard everything
        self._list_stack: list[str] = []  # each entry: "ul" or "ol"
        self._ol_counters: list[int] = []
        self._pre_depth = 0
        self._code_depth = 0
        # For each currently-open <a>, the href we will close with — or ``None`` if
        # the href was missing or javascript:, in which case we do not emit
        # markdown link syntax at all.
        self._link_href: list[str | None] = []
        self._title_parts: list[str] = []
        self._in_title = False
        # Inline text buffer for the current block; flushed on block boundaries.
        self._inline: list[str] = []

    # ---- Public accessors ----

    @property
    def title(self) -> str:
        return "".join(self._title_parts).strip()

    def get_markdown(self) -> str:
        self._flush_inline()
        text = "".join(self._out)
        lines = text.splitlines()

        # Walk lines and prefix everything between _BQ_OPEN/_BQ_CLOSE markers
        # with "> ". Nesting increases the count so a nested blockquote yields
        # "> > ".
        prefixed: list[str] = []
        bq_depth = 0
        for line in lines:
            if line == _BQ_OPEN:
                bq_depth += 1
                continue
            if line == _BQ_CLOSE:
                if bq_depth > 0:
                    bq_depth -= 1
                continue
            if bq_depth > 0:
                prefix = "> " * bq_depth
                # Blank lines inside a blockquote still get the marker so the
                # block stays visually connected in rendered markdown.
                if line == "":
                    prefixed.append(prefix.rstrip())
                else:
                    prefixed.append(f"{prefix}{line}")
            else:
                prefixed.append(line)

        # Collapse runs of blank lines and trim.
        collapsed: list[str] = []
        blank = 0
        for line in prefixed:
            if line.strip() == "":
                blank += 1
                if blank <= 1:
                    collapsed.append("")
            else:
                blank = 0
                collapsed.append(line.rstrip())
        return "\n".join(collapsed).strip() + "\n" if collapsed else ""

    # ---- HTMLParser hooks ----

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()

        if tag == "title":
            self._in_title = True
            return

        if self._drop_depth > 0:
            # Only track nesting for elements that (a) we drop and (b) have a
            # real end tag. Void tags like <input>/<source> never fire
            # handle_endtag, so counting them would leak drop-state across
            # a following close tag of the enclosing dropped element.
            if tag in _DROPPED_ELEMENTS and tag not in _VOID_ELEMENTS:
                self._drop_depth += 1
            return

        if tag in _DROPPED_ELEMENTS and tag not in _VOID_ELEMENTS:
            self._drop_depth = 1
            return
        if tag in _DROPPED_ELEMENTS:
            # Void + dropped (e.g. <input>) has no body to skip; just ignore it.
            return

        # Void elements handled here so they don't get pushed onto any stack.
        if tag == "br":
            self._inline.append("  \n")
            return
        if tag == "hr":
            self._flush_inline()
            self._out.append("\n---\n\n")
            return
        if tag == "img":
            self._handle_img(attrs)
            return

        # Block-level formatting.
        if tag in _HEADING_LEVELS:
            self._flush_inline()
            self._out.append("\n" + "#" * _HEADING_LEVELS[tag] + " ")
            return
        if tag == "p":
            self._flush_inline()
            self._out.append("\n")
            return
        if tag == "blockquote":
            self._flush_inline()
            # Emit a sentinel rather than a bare "> " prefix. A post-processing
            # pass rewrites every line between the sentinels with "> ",
            # correctly quoting nested block elements like <p> and lists.
            self._out.append(f"\n{_BQ_OPEN}\n")
            return
        if tag == "ul":
            self._flush_inline()
            self._list_stack.append("ul")
            return
        if tag == "ol":
            self._flush_inline()
            self._list_stack.append("ol")
            self._ol_counters.append(0)
            return
        if tag == "li":
            self._flush_inline()
            depth = max(0, len(self._list_stack) - 1)
            indent = "  " * depth
            if self._list_stack and self._list_stack[-1] == "ol":
                self._ol_counters[-1] += 1
                self._out.append(f"{indent}{self._ol_counters[-1]}. ")
            else:
                self._out.append(f"{indent}- ")
            return
        if tag == "pre":
            self._flush_inline()
            self._pre_depth += 1
            self._out.append("\n```\n")
            return
        if tag == "code":
            self._code_depth += 1
            if self._pre_depth == 0:
                self._inline.append("`")
            return
        if tag == "a":
            href = self._attr(attrs, "href") or ""
            if href and not href.lstrip().lower().startswith("javascript:"):
                self._link_href.append(href)
                self._inline.append("[")
            else:
                self._link_href.append(None)
            return
        if tag in ("strong", "b"):
            self._inline.append("**")
            return
        if tag in ("em", "i"):
            self._inline.append("*")
            return

        # Any other tag is treated as a transparent wrapper — its children are
        # rendered inline. We intentionally do not track it on any stack.

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag == "title":
            self._in_title = False
            return

        if self._drop_depth > 0:
            # Mirror handle_starttag exactly: only decrement on end tags of
            # elements we actually drop AND that have a matching real end tag
            # (are not void). Otherwise a stray "</input>" or "</embed>"
            # inside a dropped ancestor would decrement the counter from 1 to
            # 0 and leak the remaining dropped body.
            if tag in _DROPPED_ELEMENTS and tag not in _VOID_ELEMENTS:
                self._drop_depth -= 1
            return

        if tag in _HEADING_LEVELS:
            self._flush_inline()
            self._out.append("\n\n")
            return
        if tag == "p":
            self._flush_inline()
            self._out.append("\n\n")
            return
        if tag == "blockquote":
            self._flush_inline()
            self._out.append(f"\n{_BQ_CLOSE}\n\n")
            return
        if tag == "ul":
            self._flush_inline()
            if self._list_stack and self._list_stack[-1] == "ul":
                self._list_stack.pop()
            self._out.append("\n")
            return
        if tag == "ol":
            self._flush_inline()
            if self._list_stack and self._list_stack[-1] == "ol":
                self._list_stack.pop()
                if self._ol_counters:
                    self._ol_counters.pop()
            self._out.append("\n")
            return
        if tag == "li":
            self._flush_inline()
            self._out.append("\n")
            return
        if tag == "pre":
            self._flush_inline()
            if self._pre_depth > 0:
                self._pre_depth -= 1
            self._out.append("```\n\n")
            return
        if tag == "code":
            if self._code_depth > 0:
                self._code_depth -= 1
            if self._pre_depth == 0:
                self._inline.append("`")
            return
        if tag == "a":
            href = self._link_href.pop() if self._link_href else None
            if href is not None:
                self._inline.append(f"]({href})")
            return
        if tag in ("strong", "b"):
            self._inline.append("**")
            return
        if tag in ("em", "i"):
            self._inline.append("*")
            return

    def handle_data(self, data: str) -> None:
        if self._drop_depth > 0:
            return
        if self._in_title:
            self._title_parts.append(data)
            return
        # Discard whitespace-only chunks that appear outside of preformatted blocks
        # and outside of any current inline content — helps keep output tidy.
        if self._pre_depth > 0:
            # Preserve exact whitespace inside <pre>.
            self._out.append(data)
            return
        # Escape backticks inside inline code so they don't unbalance the ``.
        if self._code_depth > 0:
            self._inline.append(data.replace("`", ""))
            return
        # Collapse internal whitespace to single spaces to match model expectations.
        normalized = " ".join(data.split())
        if not normalized:
            # If we already have some inline text and the raw data had a space,
            # keep a single separating space.
            if data and data[0].isspace() and self._inline and not self._inline[-1].endswith(" "):
                self._inline.append(" ")
            return
        # Preserve a leading/trailing space if present in the original chunk.
        prefix = " " if data and data[0].isspace() and self._inline else ""
        suffix = " " if data and data[-1].isspace() else ""
        self._inline.append(f"{prefix}{normalized}{suffix}")

    # ---- Internals ----

    def _flush_inline(self) -> None:
        if not self._inline:
            return
        text = "".join(self._inline).strip()
        self._inline.clear()
        if text:
            self._out.append(text)

    def _handle_img(self, attrs: list[tuple[str, str | None]]) -> None:
        src = self._attr(attrs, "src") or ""
        alt = self._attr(attrs, "alt") or ""
        if not src:
            return
        # Data URIs can be enormous blobs — dropping them is a size defense as
        # well as a noise-reduction one.
        if src.lstrip().lower().startswith("data:"):
            if alt:
                self._inline.append(alt)
            return
        if src.lstrip().lower().startswith("javascript:"):
            return
        self._inline.append(f"![{alt}]({src})")

    @staticmethod
    def _attr(attrs: list[tuple[str, str | None]], name: str) -> str | None:
        name = name.lower()
        for k, v in attrs:
            if k.lower() == name:
                return unescape(v) if v is not None else None
        return None


def html_to_markdown(html: str) -> tuple[str, str]:
    """Convert HTML to markdown suitable for a model to read.

    Returns:
        A tuple of (title, markdown). Both are stripped of surrounding whitespace.
    """
    parser = _MarkdownExtractor()
    # HTMLParser tolerates malformed HTML and does not execute anything. Even
    # so, an unexpected parser bug should not fail the outer fetch -- return
    # whatever partial output has accumulated -- but leave a debug trace so a
    # real regression is visible to anyone with debug logging on.
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        logger.debug("html_to_markdown parser raised; returning partial output", exc_info=True)
    return parser.title, parser.get_markdown()
