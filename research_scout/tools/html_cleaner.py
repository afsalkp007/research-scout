"""
Zero-dependency HTML text cleaner using Python's standard library html.parser.
Avoids external dependencies while safely stripping scripts, styles, and navigation tags.
"""

from html.parser import HTMLParser
import re


class HTMLContentExtractor(HTMLParser):
    """Clean text extractor that ignores scripts, styles, and non-content elements."""

    def __init__(self):
        super().__init__()
        self._pieces = []
        self._ignore_stack = []

    def handle_starttag(self, tag, attrs):
        tag_lower = tag.lower()
        if tag_lower in ("script", "style", "nav", "footer", "header", "aside", "noscript"):
            self._ignore_stack.append(tag_lower)

    def handle_endtag(self, tag):
        tag_lower = tag.lower()
        if self._ignore_stack and self._ignore_stack[-1] == tag_lower:
            self._ignore_stack.pop()

    def handle_data(self, data):
        if not self._ignore_stack:
            clean_str = data.strip()
            if clean_str:
                self._pieces.append(clean_str)

    def get_text(self) -> str:
        raw = " ".join(self._pieces)
        return re.sub(r"\s+", " ", raw).strip()


def strip_html_tags(html_content: str) -> str:
    """Extract clean readable text from HTML markup."""
    parser = HTMLContentExtractor()
    try:
        parser.feed(html_content)
        return parser.get_text()
    except Exception:
        # Fallback regex strip if malformed markup
        clean = re.sub(r"<(script|style|nav|footer)[^>]*>.*?</\1>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r"<[^>]+>", " ", clean)
        return re.sub(r"\s+", " ", clean).strip()
