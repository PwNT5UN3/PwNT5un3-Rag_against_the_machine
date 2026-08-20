from pydantic import BaseModel, Field
import uuid


class MinimalSource(BaseModel):
    """A model for source annotations"""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Basic Question model"""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """basic question model with answer"""
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A model for datasets"""
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """A model for search results"""
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """search results but with a generated Answer"""
    answer: str


class StudentSearchResults(BaseModel):
    """A model for a result dataset"""
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """A model for a result dataset with Answers"""
    search_results: list[MinimalAnswer]
    k: int
