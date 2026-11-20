import streamlit as st

from chatbot import build_chain, format_history
from schemas import ChatResponse

CATEGORY_LABELS = {
    "programming": "💻 Programming Assistant",
    "math": "➗ Math Tutor",
    "general": "💬 General Assistant",
}

st.set_page_config(page_title="LangChain Chatbot", page_icon="🤖")
st.title("🤖 LangChain Chatbot")
st.caption("RunnableBranch routing · RunnableParallel outputs · Pydantic structured responses")


@st.cache_resource(show_spinner=False)
def get_chain():
    return build_chain()


def render_response(data: dict) -> None:
    """Display a structured response (stored as a dict) inside a chat bubble."""
    st.markdown(data["answer"])
    st.info(f"**Summary:** {data['summary']}")
    col_category, col_confidence, col_difficulty = st.columns(3)
    col_category.metric("Category", CATEGORY_LABELS[data["category"]])
    col_confidence.metric("Confidence", f"{data['confidence']:.0%}")
    col_difficulty.metric("Difficulty", data["difficulty"].capitalize())
    if data["keywords"]:
        st.markdown("**Keywords:** " + " · ".join(f"`{k}`" for k in data["keywords"]))
    if data["follow_up_questions"]:
        with st.expander("Follow-up questions"):
            for question in data["follow_up_questions"]:
                st.markdown(f"- {question}")


try:
    chain = get_chain()
except RuntimeError as error:
    st.error(str(error))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Options")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_response(message["data"])
        else:
            st.markdown(message["content"])

# User input
if user_input := st.chat_input("Ask a programming, math or general question..."):
    history = format_history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response: ChatResponse = chain.invoke(
                    {"question": user_input, "history": history}
                )
        except Exception as error:  # surface API/quota errors instead of crashing
            st.error(f"Something went wrong: {error}")
            st.session_state.messages.pop()  # drop the unanswered user message
        else:
            data = response.model_dump()
            render_response(data)
            st.session_state.messages.append(
                {"role": "assistant", "content": data["answer"], "data": data}
            )
