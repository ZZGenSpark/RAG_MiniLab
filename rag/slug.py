"""Shared slug rules for policy titles.

Both the preprocess step (markdown filenames) and the chunker (chunk ids)
turn a document title into the same slug, so a chunk id names the document
it came from and two different policies never collide.
"""

from __future__ import annotations

import re

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def slugify(title: str) -> str:
    """Turn a policy title into a lowercase, hyphenated slug.

    `Health & Wellness Policy` becomes `health-and-wellness-policy`.
    """
    lowered = title.lower().replace("&", "and")
    return _NON_ALNUM_RE.sub("-", lowered).strip("-")
