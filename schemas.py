"""Pydantic schemas used for routing, intermediate LLM outputs and the final response."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal["programming", "math", "general"]
Difficulty = Literal["beginner", "intermediate", "advanced"]


class Route(BaseModel):
    """Output of the classifier chain that feeds the RunnableBranch."""

    category: Category = Field(
        description="Best category for the user's latest question."
    )


class AnswerDraft(BaseModel):
    """Structured output produced by each branch (programming / math / general)."""

    answer: str = Field(description="Complete answer to the question, in Markdown.")
    summary: str = Field(description="A one or two sentence summary of the answer.")
    confidence: float = Field(
        description="Confidence in the answer as a number between 0.0 and 1.0."
    )


class Insights(BaseModel):
    """Structured output of the insights chain that runs in parallel with the answer."""

    keywords: list[str] = Field(description="3 to 5 key terms related to the question.")
    follow_up_questions: list[str] = Field(
        description="2 or 3 useful follow-up questions the user could ask next."
    )
    difficulty: Difficulty = Field(
        description="Difficulty level of the topic for a typical learner."
    )


class ChatResponse(BaseModel):
    """Final, validated response displayed in the Streamlit UI."""

    answer: str
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    category: Category
    keywords: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    difficulty: Difficulty

    @field_validator("confidence", mode="before")
    @classmethod
    def normalise_confidence(cls, value: float) -> float:
        """Accept 0-1 or 0-100 scales and clamp the result into [0, 1]."""
        value = float(value)
        if value > 1:
            value /= 100
        return min(max(value, 0.0), 1.0)

    @field_validator("keywords", "follow_up_questions", mode="after")
    @classmethod
    def clean_text_list(cls, items: list[str]) -> list[str]:
        """Strip whitespace, drop blanks and cap the list length."""
        return [item.strip() for item in items if item and item.strip()][:6]
