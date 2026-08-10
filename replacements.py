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
    # Hyphenated compounds (clean_word() preserves internal punctuation)
    "fuck-up": "messup",
    "fuck-all": "nothing",
    "fuck-boy": "player",
    "dumb-fuck": "idiot",
    "mother-fucker": "scoundrel",
    "mother-fucking": "flipping",
    "mind-fuck": "headgame",
    "cluster-fuck": "disaster",

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
}


def clean_word(raw: str) -> str:
    """Strip surrounding whitespace/punctuation and lowercase a word for matching.

    Examples:
        "Dang,"  -> "dang"
        "'Em"    -> "em"
        "WELL?"  -> "well"
        " hell"  -> "hell"   (faster-whisper emits leading spaces)
    Internal punctuation (e.g. "don't" -> "don't") is preserved.
    """
    import string

    return raw.strip().strip(string.punctuation).lower()


def find_matches(words: list[dict]) -> list[dict]:
    """Match a list of word-timestamp entries against the dictionary.

    Each input entry should look like:
        {"word": str, "start": float, "end": float}

    Returns a list of matched entries with the original word, its timestamps,
    and the replacement text:
        {"word": str, "start": float, "end": float, "replacement": str}
    """
    matches: list[dict] = []
    for entry in words:
        if not isinstance(entry, dict):
            continue
        raw = entry.get("word")
        if not isinstance(raw, str):
            continue
        key = clean_word(raw)
        if key in REPLACEMENTS:
            matches.append({
                "word": raw,                     # original (uncleaned) word
                "start": float(entry["start"]),
                "end": float(entry["end"]),
                "replacement": REPLACEMENTS[key],
            })
    return matches