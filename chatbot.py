import os
from operator import itemgetter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from prompts import (
    CLASSIFIER_PROMPT,
    GENERAL_PROMPT,
    INSIGHTS_PROMPT,
    MATH_PROMPT,
    PROGRAMMING_PROMPT,
)
from schemas import AnswerDraft, ChatResponse, Insights, Route

DEFAULT_MODEL = "gemini-3.1-flash-lite"


def get_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model using the API key stored in ``.env``."""
    load_dotenv(Path(__file__).parent / ".env")
    # Accept either variable name; the Gemini client reads GOOGLE_API_KEY.
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and add your "
            "Gemini API key (free key: https://aistudio.google.com/apikey)."
        )
    os.environ["GOOGLE_API_KEY"] = api_key
    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    return ChatGoogleGenerativeAI(model=model_name, max_retries=2, timeout=60)


def _assemble(data: dict[str, Any]) -> ChatResponse:
    """Merge the parallel outputs into one validated ChatResponse."""
    draft: AnswerDraft = data["draft"]
    insights: Insights = data["insights"]
    return ChatResponse(
        answer=draft.answer,
        summary=draft.summary,
        confidence=draft.confidence,
        category=data["category"],
        keywords=insights.keywords,
        follow_up_questions=insights.follow_up_questions,
        difficulty=insights.difficulty,
    )


def build_chain(llm: Any = None) -> Runnable:
    """Build the full chatbot chain.

    Input : ``{"question": str, "history": str}``
    Output: :class:`schemas.ChatResponse`

    ``llm`` can be injected (e.g. a fake model in tests); by default Gemini is used.
    """
    llm = llm or get_llm()

    # 1. Classifier: structured output -> category string used by the branch.
    classifier = (
        CLASSIFIER_PROMPT
        | llm.with_structured_output(Route)
        | RunnableLambda(lambda route: route.category)
    )

    # 2. One prompt pipeline per category, each returning a structured AnswerDraft.
    answer_llm = llm.with_structured_output(AnswerDraft)
    programming_chain = PROGRAMMING_PROMPT | answer_llm
    math_chain = MATH_PROMPT | answer_llm
    general_chain = GENERAL_PROMPT | answer_llm

    # 3. RunnableBranch: pick the pipeline that matches the category.
    #    The last positional argument is the default branch.
    branch = RunnableBranch(
        (lambda x: x["category"] == "programming", programming_chain),
        (lambda x: x["category"] == "math", math_chain),
        general_chain,
    )

    # 4. Extra outputs generated from the same request.
    insights_chain = INSIGHTS_PROMPT | llm.with_structured_output(Insights)

    # 5. RunnableParallel: run the routed answer and the insights simultaneously.
    parallel = RunnableParallel(
        draft=branch,
        insights=insights_chain,
        category=itemgetter("category"),
    )

    return (
        RunnablePassthrough.assign(category=classifier)
        | parallel
        | RunnableLambda(_assemble)
    )


def format_history(messages: list[dict[str, str]], max_messages: int = 6) -> str:
    """Turn the most recent chat messages into a plain-text history for prompts."""
    lines = []
    for message in messages[-max_messages:]:
        speaker = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {message['content']}")
    return "\n".join(lines) or "(no previous messages)"
