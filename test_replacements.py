#!/usr/bin/env python
"""Regression tests for the matching logic in replacements.py.

Guards the three normalization behaviors documented in handoff notes:
curly-quote normalization, possessive ('s) stem matching, and hyphen
fallbacks. Run directly (python test_replacements.py) or via pytest.

The MUST-NOT-MATCH set is as important as the matches: it pins the
exact-key philosophy ("assume" and "embarrassing" contain "ass" but are
clean words; "don't" must keep its internal apostrophe; glued plurals
like "bitchs" must not be silently remapped).
"""
from replacements import clean_word, find_matches

# raw token -> expected replacement
MATCH_CASES = {
    # exact keys
    "ass": "butt",
    "Dumbass,": "idiot",
    "God": "gosh",
    "kick-ass": "kickbutt",       # curated hyphenated key beats fallbacks
    "half-assed": "sloppy",
    # curly quotes normalized before matching
    "God\u2019s": "gosh's",
    # possessive stem retry
    "shit's": "crap's",
    "bitch's": "brat's",
    # hyphen fallback 2: hyphens stripped
    "dumb-ass": "idiot",
    "dumb-ass's": "idiot's",
    # hyphen fallback 3: per-part substitution
    "weak-ass": "weak-butt",
    "monkey-ass": "monkey-butt",
    "ass-in-ass": "butt-in-butt",
    "stupid-as-shit": "stupid-as-crap",
    # motherfucker variants Whisper emits besides the plain forms
    "Motherfuck!": "scoundrel",
    "Motherfuckin'": "flippin",
    "Motherf-": "scoundrel",
    # Whisper's literal asterisk spellings (audio says the full word)
    "f***ing": "stinking",
    "f**ked.": "freaked",
    "F**k?": "freak",
    "f***": "freak",        # asterisk-final form survives clean_word()
    "f***.": "freak",
    # glued fuck-compounds
    "fucknuts.": "twits",
    "DERPFUCKER!": "doofus",
}

NO_MATCH_CASES = [
    "assume",           # "ass" substring, clean word
    "embarrassing.",
    "don't",            # internal apostrophe preserved, not a key
    "mother-in-law",    # no profane part
    "t-shirt",
    "bitchs",           # glued plural is deliberately not a key
    "f-",               # cut-off letter F; audio never completes the word
    "F'd",              # softened "effed" — audio already says a clean form
    "f-ing",            # softened "eff-ing"
    "self-aware",       # clean hyphenated compounds
    "",
]


def _one(word):
    return find_matches([{"word": word, "start": 0.0, "end": 1.0}])


def test_match_cases():
    for raw, expected in MATCH_CASES.items():
        matches = _one(raw)
        assert len(matches) == 1, f"{raw!r}: expected 1 match, got {matches}"
        assert matches[0]["replacement"] == expected, (
            f"{raw!r}: expected {expected!r}, got {matches[0]['replacement']!r}")
        assert matches[0]["word"] == raw  # original token preserved


def test_no_match_cases():
    for raw in NO_MATCH_CASES:
        assert _one(raw) == [], f"{raw!r}: should not match"


def test_clean_word():
    assert clean_word(" hell") == "hell"
    assert clean_word("'Em") == "em"
    assert clean_word("WELL?") == "well"
    assert clean_word("don\u2019t") == "don't"


if __name__ == "__main__":
    test_match_cases()
    test_no_match_cases()
    test_clean_word()
    print(f"all tests passed ({len(MATCH_CASES)} match, "
          f"{len(NO_MATCH_CASES)} no-match, 4 clean_word)")
