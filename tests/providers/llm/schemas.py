"""Pydantic schemas used by the adapter tests."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Verdict(BaseModel):
    """A fact-check verdict for one claim."""

    label: Literal["verified", "wrong", "unsupported"]
    confidence: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)


class Source(BaseModel):
    url: str


class Cited(BaseModel):
    """Nested model to exercise reference inlining."""

    sources: list[Source]
    primary: Source | None = None


class Tree(BaseModel):
    """Recursive model: $defs must be kept."""

    name: str
    children: list[Tree] = Field(default_factory=list)
