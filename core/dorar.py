"""Lookup in Dorar al-Saniyya (dorar.net/hadith), the hadith source named in
the challenge's scientific package.

Used only for texts that were NOT found in the two Sahihs. Dorar returns the
narrations it holds with the grading of each scholar (the "خلاصة حكم
المحدث"). The tool shows those gradings with the scholar's name and never
produces a grading of its own.

Dorar's public endpoint returns JSON with an HTML fragment inside:
    https://dorar.net/dorar_api.json?skey=<text>
If the site is unreachable the lookup fails softly and the report falls back
to "refer to a specialist".
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

API_URL = "https://dorar.net/dorar_api.json"
TIMEOUT = 4
# After a failure, skip Dorar for a while so the page stays fast.
RETRY_AFTER = 300
_down_until = 0.0

_FIELDS = {
    "الراوي": "narrator",
    "المحدث": "scholar",
    "المصدر": "source",
    "الصفحة أو الرقم": "number",
    "خلاصة حكم المحدث": "grade",
}


@dataclass
class DorarEntry:
    text: str
    narrator: str = ""
    scholar: str = ""
    source: str = ""
    number: str = ""
    grade: str = ""


def parse(html: str) -> list[DorarEntry]:
    """Turn Dorar's HTML fragment into entries."""
    soup = BeautifulSoup(html, "html.parser")
    entries: list[DorarEntry] = []
    for info in soup.select(".hadith-info"):
        body = info.find_previous_sibling(class_="hadith")
        text = body.get_text(" ", strip=True) if body else ""
        text = re.sub(r"^\d+\s*-\s*", "", text)  # drop the "1 - " counter
        entry = DorarEntry(text=text)
        for label in info.select(".info-subtitle"):
            key = label.get_text(strip=True).rstrip(":").strip()
            attr = _FIELDS.get(key)
            if not attr:
                continue
            # The value is the text right after the label, up to the next label.
            value_parts = []
            for sib in label.next_siblings:
                if getattr(sib, "get", None) and "info-subtitle" in (sib.get("class") or []):
                    break
                value_parts.append(sib.get_text(" ", strip=True) if hasattr(sib, "get_text") else str(sib).strip())
            setattr(entry, attr, " ".join(p for p in value_parts if p).strip(" |"))
        entries.append(entry)
    return entries


def lookup(text: str) -> list[DorarEntry] | None:
    """Search Dorar. Returns None when the site cannot be reached."""
    global _down_until
    if time.time() < _down_until:
        return None
    try:
        r = requests.get(API_URL, params={"skey": text}, timeout=TIMEOUT,
                         headers={"User-Agent": "Mozilla/5.0 (Tathabbat hadith checker)"})
        r.raise_for_status()
        html = r.json().get("ahadith", {}).get("result", "")
    except (requests.RequestException, ValueError):
        _down_until = time.time() + RETRY_AFTER
        return None
    return parse(html)
