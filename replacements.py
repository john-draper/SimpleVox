#!/usr/bin/env python
"""
Profanity replacement dictionary for SimpleVox.

This module holds the hardcoded mapping of profane words -> clean euphemisms.
It is intentionally kept separate from the matching logic so it is easy to
edit, and so other modules (and tests) can import it directly.

NOTE ON THE "BITCH" REPLACEMENT
-------------------------------
A previous version of this dictionary mapped the word to "bench". That is a
poor choice: "bench" is near-homophonous with the original word (same starting
consonant cluster "b-ch", same short vowel shape), so the censored audio still
sounds like the profanity. We now map it to "brat" instead, which has:
  - a completely different vowel,
  - a different ending consonant ("t" vs the affricate "ch"),
  - a different number of syllables when spoken quickly,
making the substitution genuinely unrecognizable as the original.
"""

# --------------------------------------------------------------------------- #
# Hardcoded replacement dictionary.
#   key   = cleaned word (lowercase, no surrounding punctuation) to match
#   value = replacement text to substitute
# Edit this freely — add or remove mappings as needed.
# --------------------------------------------------------------------------- #
#
# This list was seeded from an "Advanced Profanity Filter" export and trimmed
# to single-word entries. The matching is case-insensitive and ignores
# surrounding punctuation (see clean_word()).
#
# NOTE: Only actual profanity belongs here. Clean euphemisms (dang, darn, gosh,
# heck, shucks) are the REPLACEMENT targets, not words to filter.
# --------------------------------------------------------------------------- #
REPLACEMENTS: dict[str, str] = {
    # --- Ass family ---
    "ass": "butt",
    "asses": "butts",
    "asshole": "jerk",
    "assholes": "jerks",
    "assholimov": "buttholimov",
    "assholishness": "buttholishness",
    "assing": "butting",
    "asswipe": "fool",
    "ass-wipe": "fool",
    "asshat": "jerk",
    "assclown": "goofball",
    "assface": "butthead",
    "badass": "cool",
    "badasses": "cool guys",
    "dumbass": "idiot",
    "dumbasses": "idiots",
    "jackass": "jerk",
    "jackasses": "jerks",
    "kickass": "kickbutt",
    "kick-ass": "kickbutt",
    "smartass": "smarty",
    "smartasses": "smarties",
    # Hyphenated compounds (clean_word() preserves internal punctuation)
    "half-assed": "sloppy",
    "half-ass": "lazy",
    "lazy-ass": "lazy",
    "cheap-ass": "cheap",
    "hard-ass": "stickler",
    "big-ass": "huge",
    "bad-ass": "tough",

    # --- Shit family ---
    "apeshit": "apecrap",
    "batshit": "batcrap",
    "birdshit": "birdcrap",
    "bullshit": "bull",
    "chickenshit": "chickencrap",
    "dogshit": "dogcrap",
    "dumbshit": "dumbcrap",
    "horseshit": "horsecrap",
    "shit": "crap",
    "shitass": "crapbutt",
    "shitbag": "crapbag",
    "shitbeard": "crapbeard",
    "shitbird": "crapbird",
    "shitbox": "crapbox",
    "shitbrain": "crapbrain",
    "shitbrains": "crapbrains",
    "shitface": "crapface",
    "shitfaced": "crapfaced",
    "shithead": "craphead",
    "shitheads": "crapheads",
    "shitheel": "crapheel",
    "shithole": "craphole",
    "shitlick": "craplick",
    "shitload": "crapload",
    "shitloads": "craploads",
    "shits": "craps",
    "shitsnackin": "crapsnackin",
    "shitsnacks": "crapsnacks",
    "shitspace": "crapspace",
    "shitstain": "crudstain",
    "shitstains": "crudstains",
    "shitstorm": "crapstorm",
    "shitstorms": "crapstorms",
    "shitshow": "mess",
    "shitter": "crapper",
    "shittier": "crappier",
    "shittiest": "crappiest",
    "shittin": "crappin",
    "shitting": "crapping",
    "shitty": "crappy",
    "shitzombies": "crapzombies",
    "shart": "poo-fart",
    # Bullshit variants
    "bullshitter": "liar",
    "bullshitters": "liars",
    "bullshitting": "fibbing",
    "bullshitted": "fibbed",
    # Dipshit variants
    "dipshits": "dipsticks",
    # Hyphenated compounds (clean_word() preserves internal punctuation)
    "jackshit": "nothing",
    "jack-shit": "nothing",
    "dip-shit": "dipstick",
    "shit-ton": "ton",

    # --- Fuck family ---
    "fuck": "freak",
    "fucks": "freaks",
    "fucker": "jerk",
    "fuckers": "jerks",
    "fucked": "freaked",
    "fucking": "stinking",  # "(beep)ing" / "f***ing" -> "stinking"
    "fuckin": "freakin",
    "fuckface": "jerk",
    "fuckhead": "jerk",
    "fuckboy": "player",
    "fuckboys": "players",
    "fucktard": "idiot",
    "fuckwit": "idiot",
    "fuckwits": "idiots",
    "fuckup": "messup",
    "fuckall": "nothing",
    "dumbfuck": "idiot",
    "dumbfucks": "idiots",
    "motherfucker": "scoundrel",
    "motherfuckers": "scoundrels",
    "motherfucking": "flipping",
    "mindfuck": "headgame",
    "clusterfuck": "disaster",
    "effing": "flipping",
    # Motherfucker variants Whisper emits that aren't the plain forms:
    # "Motherfuck!", "Motherfuckin'", and the cut-off "Motherf-".
    "motherfuck": "scoundrel",
    "motherfucks": "scoundrels",
    "motherfuckin": "flippin",
    "motherf": "scoundrel",
    # Glued fuck-compounds seen in transcripts (no hyphen -> no fallback).
    "fucknut": "twit",
    "fucknuts": "twits",
    "fuckpot": "weirdo",
    "derpfucker": "doofus",
    # Hyphenated compounds (clean_word() preserves internal punctuation)
    "fuck-up": "messup",
    "fuck-all": "nothing",
    "fuck-boy": "player",
    "dumb-fuck": "idiot",
    "mother-fucker": "scoundrel",
    "mother-fucking": "flipping",
    "mind-fuck": "headgame",
    "cluster-fuck": "disaster",

    # Whisper sometimes self-censors and emits literal asterisk spellings
    # ("f***ing", "f**ked") while the AUDIO says the full word. These are
    # audible profanity, so they are keys like any other form. clean_word()
    # strips the surrounding punctuation, so keys are the bare asterisk forms.
    "f***": "freak",
    "f***s": "freaks",
    "f***er": "jerk",
    "f***ing": "stinking",
    "f***ed": "freaked",
    "f**k": "freak",
    "f**ker": "jerk",
    "f**king": "stinking",
    "f**ked": "freaked",

    # --- Damn / hell / goddamn family ---
    "damn": "dang",
    "damns": "dangs",
    "damned": "danged",
    "damning": "danging",
    "dammit": "dangit",
    "god": "gosh",
    "goddamn": "doggone",
    "goddamned": "doggone",
    "goddamns": "doggones",
    "goddamning": "doggoning",
    "goddammit": "dangit",
    "hell": "heck",

    # --- Religious exclamations (used as profanity) ---
    "christ": "cripes",
    "christs": "cripes",
    "jesus": "geez",

    # --- Bitch / cunt / twat family ---
    # NOTE: "bitch" -> "brat" (previously "bench", which sounded too similar).
    "bitch": "brat",
    "bitches": "brats",
    "bitching": "complaining",
    "bitchy": "bratty",
    "bitchier": "brattier",
    "bitchiest": "brattiest",
    "bitchslap": "smack",
    "bitchfest": "gripefest",
    # Slang spellings
    "biatch": "brat",
    "biznatch": "brat",
    "betch": "brat",
    # Hyphenated compounds (clean_word() preserves internal punctuation)
    "bitch-slap": "smack",
    "bitch-fest": "gripefest",
    "cunt": "expletive",
    "twat": "dumbo",
    "twats": "dumbos",

    # --- Cocksucker ---
    "cocksucker": "suckup",

    # --- Pussy / pussies ---
    "pussy": "softie",
    "pussies": "softies",

    # --- Misc from the filter ---
    "bleep": "beep",
    "fags": "gays",
    "fuchs": "craps",
    "dipshit": "dipstick",

    # --- Douche family ---
    "douche": "jerk",
    "douches": "jerks",
    "douchebag": "ninny",
    "douchebags": "ninnies",
    "douche-canoe": "ninny",

    # --- Bastard family ---
    "bastard": "rascal",
    "bastards": "rascals",
    "bastardy": "rascally",

    # --- Prick family ---
    "prick": "twit",
    "pricks": "twits",

    # --- Dick family ---
    "dick": "dork",
    "dicks": "dorks",
    "dickhead": "doofus",
    "dickheads": "doofuses",
    "dickwad": "doofus",
    "dickhole": "doofus",

    # --- Piss family ---
    "piss": "pee",
    "pisses": "pees",
    "pissed": "ticked",
    "pissing": "ticking",
    "pisser": "stinker",
    "pissed-off": "ticked-off",

    # --- Retard family ---
    "retard": "dummy",
    "retards": "dummies",
    "retarded": "ridiculous",

    # --- Cock family ---
    "cock": "willy",
    "cocks": "willies",

    # --- Whore / slut ---
    "whore": "hussy",
    "whores": "hussies",
    # "slut"/"sluts"/"slutty" REMOVED 2026-10-07 at the owner's request:
    # slut-family words are deliberately NOT censored. Do not re-add them
    # (also covers the hyphen per-part fallback - "X-slut" stays unmatched).

    # --- Small gaps ---
    "goddamnit": "doggone",
    "hells": "hecks",
}


def clean_word(raw: str) -> str:
    """Strip surrounding whitespace/punctuation and lowercase a word for matching.

    Examples:
        "Dang,"  -> "dang"
        "'Em"    -> "em"
        "WELL?"  -> "well"
        " hell"  -> "hell"   (faster-whisper emits leading spaces)
    Internal punctuation (e.g. "don't" -> "don't") is preserved.
    Curly quotes/apostrophes are normalized to ASCII first (Whisper
    occasionally emits them; string.punctuation does not include them).
    """
    import string

    normalized = (raw.replace("\u2019", "'")   # right single quote
                     .replace("\u2018", "'")   # left single quote
                     .replace("\u201c", '"')   # left double quote
                     .replace("\u201d", '"'))  # right double quote
    return normalized.strip().strip(string.punctuation).lower()


def _resolve_key(key: str) -> str | None:
    """Resolve a cleaned token to a replacement, with hyphen fallbacks.

    Whisper routinely emits hyphenated compounds ("dumb-ass", "weak-ass",
    "ass-in-ass") whose exact form is not a dictionary key. Resolution order:

      1. exact key — curated compounds always win ("kick-ass" -> "kickbutt",
         "half-assed" -> "sloppy")
      2. hyphens stripped — "dumb-ass" -> "dumbass" -> "idiot"
      3. per-part — replace only the profane hyphen-parts, keep the rest
         verbatim ("weak-ass" -> "weak-butt", "ass-in-ass" -> "butt-in-butt",
         "monkey-ass" -> "monkey-butt"); fires only if at least one part
         is a key, so clean compounds ("mother-in-law") never match

    clean_word() preserves internal punctuation, so this hyphen handling
    must live HERE (in matching) — do not move it into clean_word().
    """
    if key in REPLACEMENTS:
        return REPLACEMENTS[key]
    if "-" not in key:
        return None
    glued = key.replace("-", "")
    if glued in REPLACEMENTS:
        return REPLACEMENTS[glued]
    parts = key.split("-")
    if any(p in REPLACEMENTS for p in parts):
        return "-".join(REPLACEMENTS.get(p, p) for p in parts)
    return None


def find_matches(words: list[dict]) -> list[dict]:
    """Match a list of word-timestamp entries against the dictionary.

    Each input entry should look like:
        {"word": str, "start": float, "end": float}

    Returns a list of matched entries with the original word, its timestamps,
    and the replacement text:
        {"word": str, "start": float, "end": float, "replacement": str}

    Possessives/contractions ("shit's", "bitch's", "God's") match on the stem
    ("shit", "bitch", "god") with "'s" appended to the spoken replacement
    ("crap's", "brat's", "gosh's"). The stem is guaranteed to be a dictionary
    key (it heads every word family), whereas the glued plural produced by
    deleting the apostrophe ("bitchs") often is not. The stem itself goes
    through _resolve_key, so possessive hyphen compounds also work
    ("dumb-ass's" -> "idiot's").
    """
    matches: list[dict] = []
    for entry in words:
        if not isinstance(entry, dict):
            continue
        raw = entry.get("word")
        if not isinstance(raw, str):
            continue
        key = clean_word(raw)
        replacement = _resolve_key(key)
        if replacement is None and key.endswith("'s"):
            stem_replacement = _resolve_key(key[:-2])
            if stem_replacement is not None:
                replacement = stem_replacement + "'s"
        if replacement is None and "*" in raw:
            # Whisper's censored spellings that END in asterisks ("f***"):
            # clean_word() strips trailing asterisks as punctuation, leaving
            # a bare "f" that matches nothing. Strip every punctuation
            # character EXCEPT asterisks, so the censored form itself is the
            # lookup key ("f***" -> dictionary entry "f***").
            import string
            star_key = raw.strip().strip(
                "".join(c for c in string.punctuation if c != "*")).lower()
            replacement = REPLACEMENTS.get(star_key)
        if replacement is not None:
            matches.append({
                "word": raw,                     # original (uncleaned) word
                "start": float(entry["start"]),
                "end": float(entry["end"]),
                "replacement": replacement,
            })
    return matches