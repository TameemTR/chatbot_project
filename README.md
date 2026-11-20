# LangChain Chatbot — RunnableBranch & RunnableParallel

A Streamlit chatbot built with the latest LangChain APIs and Google Gemini. Every question is
classified, routed to a specialised prompt pipeline with **RunnableBranch**, enriched with extra
outputs generated concurrently by **RunnableParallel**, and returned as a validated
**Pydantic** structured response.

## Features

- Chat interface with history, input box and a **Clear chat** button (Streamlit)
- Automatic routing: Programming Assistant, Math Tutor or General Assistant
- Parallel generation of the answer and extra insights (keywords, follow-up questions, difficulty)
- Structured, validated output: answer, summary, confidence, category, keywords, follow-ups, difficulty
- All prompts built with `PromptTemplate` (none inside `invoke()`)
- API key stored in `.env` (never committed)
- Works with the Gemini API free tier

## Project structure

```
chatbot/
├── app.py            # Streamlit UI
├── chatbot.py        # Chain construction (branch, parallel, assembly)
├── prompts.py        # PromptTemplates
├── schemas.py        # Pydantic models
├── tests/            # Offline tests (fake LLM, no API key needed)
├── assets/           # Screenshots (optional)
├── requirements.txt
├── .env.example
└── README.md
```

## How it works

```
User question
      │
      ▼
 Classifier (Pydantic Route)
      │
      ▼
 RunnableParallel ───────────────┬──────────────────────┐
      │                          │                      │
 RunnableBranch            Insights chain          category
 ├─ Programming prompt     (keywords, follow-ups,  (passthrough)
 ├─ Math prompt             difficulty)
 └─ General prompt (default)
      │                          │                      │
      └──────────────────────────┴──────────────────────┘
                                 ▼
                 ChatResponse (Pydantic validation)
                                 ▼
                         Streamlit chat UI
```

### RunnableBranch implementation

A small classifier chain returns a category (`programming`, `math` or `general`). The category is
added to the input with `RunnablePassthrough.assign`, and `RunnableBranch` picks the matching
prompt pipeline. The last argument is the default branch (`general`).

```python
branch = RunnableBranch(
    (lambda x: x["category"] == "programming", programming_chain),
    (lambda x: x["category"] == "math", math_chain),
    general_chain,
)
```

### RunnableParallel implementation

The routed answer chain and the insights chain run at the same time on the same input.

```python
parallel = RunnableParallel(
    draft=branch,                      # answer + summary + confidence
    insights=insights_chain,           # keywords + follow-ups + difficulty
    category=itemgetter("category"),
)
```

### Pydantic Structured Output implementation

`schemas.py` defines `Route`, `AnswerDraft`, `Insights` and the final `ChatResponse`.
Each LLM step uses `llm.with_structured_output(<Model>)`, so the model returns validated objects
instead of free text. `ChatResponse` adds validators (confidence normalised to 0–1, lists
cleaned and capped) and the Streamlit UI renders its fields.

## Installation

Requires Python 3.10 or newer.

```bash
cd chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GEMINI_API_KEY (free key: https://aistudio.google.com/apikey)

streamlit run app.py
```

To change the model, set `GEMINI_MODEL` in `.env` (default: `gemini-3.1-flash-lite`, which is
light and reliable on the free tier). Model availability changes over time; if you get a
`404 NOT_FOUND` or `503` error, pick another model from https://aistudio.google.com.

### Run the tests

```bash
pytest
```

The tests use a fake LLM, so they run offline without an API key.

## Notes

- The free Gemini tier has per-minute and per-day request limits. Each message makes three model
  calls (classifier, answer, insights), so heavy use may hit rate limits.
- Never commit `.env` or any API key.
