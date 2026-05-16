from __future__ import annotations

import html
import re


URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
RT_PREFIX_RE = re.compile(r"^\s*(?:RT\s+@[A-Za-z0-9_]{1,20}:?\s*)+", flags=re.IGNORECASE)
HASHTAG_SIGN_RE = re.compile(r"#")
GREEK_EXTENDED_RE = re.compile(r"[\u1F00-\u1FFF]+")
CONTROL_CHAR_RE = re.compile(r"[\u0000-\u0008\u000B-\u001F\u007F-\u009F]+")

MOJIBAKE_MAP = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "‚Ä¶": "...",
    "‚Äù": '"',
    "‚Äú": '"',
    "‚Äô": "'",
    "Â": "",
}


def clean_tweet_text(tweet_text: str) -> str:
    text = html.unescape(tweet_text or "")

    for bad, good in MOJIBAKE_MAP.items():
        if bad in text:
            text = text.replace(bad, good)

    if re.search(r"[âÃÂ‚á¼á½]", text):
        try:
            repaired = text.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore")
            if repaired:
                text = repaired
        except Exception:
            pass

    text = GREEK_EXTENDED_RE.sub("", text)
    text = CONTROL_CHAR_RE.sub(" ", text)
    text = URL_RE.sub("", text)
    text = RT_PREFIX_RE.sub("", text)
    text = HASHTAG_SIGN_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

