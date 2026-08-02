import re


_QUESTION_WORDS = ["what", "is", "are", "how", "does", "do", "why", "when",
    "where", "which", "who", "the", "a", "an", "in", "of",
    "for", "to", "and", "explain", "describe", "tell", "me", "about", "i", "you"]

def streamline_query(query: str):
    query = re.sub(r"[^\w\s]", " ", query)
    words = query.lower().split(" ")
    query = " ".join([word for word in words if word not in _QUESTION_WORDS])
    return query

def clean_text_chunks(chunk: str):
    chunk = re.sub(r"<[^>]+>", " ", chunk)
    chunk = re.sub(r"```[\w+-]*\n(.*?)```", r"\n\1\n", chunk, flags=re.S)
    chunk = re.sub(r"`([^`]*)`", r"\n\1\n", chunk)
    chunk = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", chunk)
    chunk = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", chunk)
    chunk = re.sub(r"\*\*([^*]+)\*\*", r"\1", chunk)
    chunk = re.sub(r"\*([^*]+)\*", r"\1", chunk)
    chunk = re.sub(r"__([^_]+)__", r"\1", chunk)
    chunk = re.sub(r"_([^_]+)_", r"\1", chunk)
    chunk = re.sub(r"^\s{0,3}#{1,6}\s*", "", chunk, flags=re.M)
    chunk = re.sub(r"^\s*[-*+]\s+", "", chunk, flags=re.M)
    chunk = re.sub(r"^\s*>\s?", "", chunk, flags=re.M)
    chunk = re.sub(r"^\s*([-*_]\s*){3,}\s*$", " ", chunk, flags=re.M)
    chunk = re.sub(r"[~^]", " ", chunk)
    chunk = re.sub(r"\s+", " ", chunk).strip()
    return chunk.encode("ascii", "ignore").decode("ascii")