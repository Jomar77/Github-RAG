import asyncio

import streamlit as st

import ingest
import logs
import search_agent


REPO_OWNER = "DataTalksClub"
REPO_NAME = "faq"


@st.cache_resource(show_spinner="Building search index...")
def get_index():
    def filter_doc(doc):
        return "data-engineering" in doc["filename"]

    return ingest.index_data(REPO_OWNER, REPO_NAME, filter=filter_doc)


@st.cache_resource(show_spinner="Initializing agent...")
def get_agent(_index):
    return search_agent.init_agent(_index, REPO_OWNER, REPO_NAME)


def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # Fallback for environments that already hold an event loop.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def init_state():
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "model_message_history" not in st.session_state:
        st.session_state.model_message_history = []
    if "history_supported" not in st.session_state:
        st.session_state.history_supported = True


def clear_chat():
    st.session_state.chat_messages = []
    st.session_state.model_message_history = []
    st.session_state.history_supported = True


def run_agent(agent, question):
    if st.session_state.history_supported:
        try:
            return run_async(
                agent.run(
                    user_prompt=question,
                    message_history=st.session_state.model_message_history,
                )
            )
        except TypeError as exc:
            if "message_history" not in str(exc):
                raise
            st.session_state.history_supported = False
            st.warning(
                "Message history is not supported by the installed pydantic-ai API. "
                "Continuing in single-turn mode."
            )

    return run_async(agent.run(user_prompt=question))


def main():
    st.set_page_config(page_title="DataTalks FAQ Assistant", layout="wide")
    st.title("DataTalks FAQ Assistant")
    st.caption(f"Documentation source: {REPO_OWNER}/{REPO_NAME}")

    init_state()

    with st.sidebar:
        st.subheader("Session")
        if st.button("Clear chat", use_container_width=True):
            clear_chat()
            st.rerun()

    try:
        index = get_index()
        agent = get_agent(index)
    except Exception as exc:
        st.error(f"Initialization failed: {exc}")
        return

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about the documentation...")
    if not question:
        return

    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and answering..."):
            try:
                response = run_agent(agent, question)
                answer = str(response.output)
                st.markdown(answer)

                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": answer}
                )

                new_messages = list(response.new_messages())
                if st.session_state.history_supported:
                    st.session_state.model_message_history.extend(new_messages)

                logs.log_interaction_to_file(agent, new_messages)
            except Exception as exc:
                st.error(f"Request failed: {exc}")


if __name__ == "__main__":
    main()
