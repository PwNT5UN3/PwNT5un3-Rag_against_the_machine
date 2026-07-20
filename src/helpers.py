import re


_QUESTION_WORDS = []

def streamline_query(query: str):
    query = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", query)
    query = re.sub(r"[^\w\s]", " ", query)
    words = query.lower().split(" ")
    query = " ".join([word for word in words if word not in _QUESTION_WORDS])
    return query