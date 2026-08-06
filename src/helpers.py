import re

_QUESTION_WORDS = [
    "what",
    "is",
    "are",
    "how",
    "does",
    "do",
    "why",
    "when",
    "where",
    "which",
    "who",
    "the",
    "a",
    "an",
    "in",
    "of",
    "for",
    "to",
    "and",
    "explain",
    "describe",
    "tell",
    "me",
    "about",
    "i",
    "you",
    "used",
    "using",
    "wether",
]


def streamline_query(query: str):
    query = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", query).lower()
    query = re.sub(r"[^\w\s]|(?<!\w)_|_(?!\w)", " ", query)
    words = query.lower().split(" ")
    if "constructor" in words:
        words.append("init")
    query = " ".join(
        [
            word
            for word in words
            if word not in _QUESTION_WORDS and len(word) >= 2
        ]
    )
    print(query)
    return query


def clean_text_chunks(chunk: str):
    chunk = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", chunk).lower()
    chunk = re.sub(r"[^\w\s]|(?<!\w)_|_(?!\w)", " ", chunk)
    tokens = chunk.split()
    return " ".join([t for t in tokens if len(t) >= 1])
