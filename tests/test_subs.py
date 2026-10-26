#!/usr/bin/env python
"""Unit tests for subtitle rewriting (subs.py). No media files required."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from subs import rewrite_srt, rewrite_text  # noqa: E402


def check(text, want):
    got, hits = rewrite_text(text)
    assert got == want, f"{text!r}: got {got!r}, want {want!r}"
    return hits


def main():
    n = 0
    n += check("Damn it.", "Dang it.")
    n += check("I swear to God.", "I swear to Gosh.")
    n += check("GOD, no!", "GOSH, no!")
    n += check("god's plan", "gosh's plan")
    n += check("What the fuck?", "What the freak?")
    n += check("f***ing hell", "stinking heck")
    n += check("Christ!", "Sheesh!")
    n += check("Oh, Christ.", "Oh, Sheesh.")
    n += check("that dumb-ass move", "that idiot move")
    n += check("kick-ass show", "kickbutt show")
    n += check("classic glass", "classic glass")  # no substring matches
    assert n == 11, f"expected 11 hits, got {n}"

    srt = ("1\n00:00:01,000 --> 00:00:02,000\nDamn it.\n\n"
           "2\n00:00:03,000 --> 00:00:04,000\n<i>Shit.</i>\n")
    out, hits = rewrite_srt(srt)
    assert hits == 2, hits
    assert "00:00:01,000 --> 00:00:02,000" in out
    assert "Dang it." in out and "<i>Crap.</i>" in out
    assert "Damn" not in out and "Shit" not in out
    print("PASS: subtitle rewrite unit tests")


if __name__ == "__main__":
    main()
