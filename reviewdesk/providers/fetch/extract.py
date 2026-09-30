"""Readable-text extraction from HTML with BeautifulSoup (``html.parser``).

Removes scripts, styles, navigation chrome, forms and hidden elements, then
prefers the main content container (``<article>``, ``<main>``,
``role=main``) when it holds most of the text. Block elements become line
breaks; whitespace inside a line is collapsed.

The output is untrusted page content. It is only extracted here, never
interpreted; hidden elements are dropped partly because they are a common
place to hide prompt-injection text.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag
from bs4.element import Comment, Declaration, Doctype, ProcessingInstruction

DROP_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "iframe",
    "object",
    "embed",
    "form",
    "button",
    "select",
    "input",
    "textarea",
    "nav",
    "header",
    "footer",
    "aside",
    "head",
)
BLOCK_TAGS = (
    "p",
    "div",
    "section",
    "article",
    "main",
    "li",
    "ul",
    "ol",
    "dl",
    "dt",
    "dd",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
    "pre",
    "table",
    "tr",
    "figure",
    "figcaption",
    "hr",
    "address",
    "details",
    "summary",
)
CELL_TAGS = ("td", "th")
MAIN_SELECTORS = ("article", "main", "[role=main]")
MAIN_SHARE = 0.4
"""The main container is used if it holds at least this share of body text."""

_SPACE_RE = re.compile(r"[ \t\r\f\v ​]+")
_HIDDEN_STYLE_RE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.IGNORECASE)


def _is_hidden(tag: Tag) -> bool:
    if tag.attrs is None:
        return False
    if tag.has_attr("hidden"):
        return True
    if str(tag.get("aria-hidden", "")).lower() == "true":
        return True
    style = tag.get("style")
    return isinstance(style, str) and bool(_HIDDEN_STYLE_RE.search(style))


def _title(soup: BeautifulSoup) -> str:
    og = soup.find("meta", attrs={"property": "og:title"})
    if isinstance(og, Tag):
        content = og.get("content")
        if isinstance(content, str) and content.strip():
            return _SPACE_RE.sub(" ", content).strip()
    if soup.title is not None and soup.title.string:
        return _SPACE_RE.sub(" ", soup.title.string).strip()
    h1 = soup.find("h1")
    if isinstance(h1, Tag):
        return _SPACE_RE.sub(" ", h1.get_text(" ")).strip()
    return ""


def _normalize(text: str) -> str:
    lines = (_SPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def _text_of(node: Tag) -> str:
    return _normalize(node.get_text())


def extract_readable(markup: str | bytes, *, encoding: str | None = None) -> tuple[str, str]:
    """Return ``(title, text)`` for an HTML document."""
    from_encoding = encoding if isinstance(markup, bytes) else None
    soup = BeautifulSoup(markup, "html.parser", from_encoding=from_encoding)
    title = _title(soup)

    for node in soup.find_all(
        string=lambda s: isinstance(s, Comment | Declaration | Doctype | ProcessingInstruction)
    ):
        node.extract()
    for tag in soup.find_all(DROP_TAGS):
        if not tag.decomposed:
            tag.decompose()
    for tag in soup.find_all(_is_hidden):
        if not tag.decomposed:
            tag.decompose()

    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    for tag in soup.find_all(CELL_TAGS):
        tag.insert_after(" ")

    root: Tag = soup.body if soup.body is not None else soup
    full = _text_of(root)
    for selector in MAIN_SELECTORS:
        candidates = root.select(selector)
        if not candidates:
            continue
        best = max(candidates, key=lambda c: len(c.get_text()))
        main = _text_of(best)
        if full and len(main) >= MAIN_SHARE * len(full):
            return title, main
    return title, full


def plain_text(text: str) -> str:
    """Normalize a ``text/plain`` body the same way as extracted HTML."""
    return _normalize(text)
