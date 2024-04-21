import time
import streamlit as st
from src.pdf_parser import (
    get_pdf_text,
    get_text_chunks
)
from src.llm_utils import (
    get_vector_store,
    user_input,
)
from src.chat_utils import response_generator


def main():
    st.set_page_config(
        page_title="ChatAbraham",
        initial_sidebar_state="collapsed",
        )
    st.header("ChatAbraham")
    with st.sidebar:
        if st.button("Configure App"):
            with st.spinner("Configuring App..."):
                pdf_text = get_pdf_text("./data")
                text_chunks = get_text_chunks(pdf_text)
                get_vector_store(text_chunks)
                st.success("App Configured!")

    if st.button("Clear Chat History"):
        st.session_state["messages"] = []
        st.session_state["chat_history"] = []

    user_question = st.chat_input(
        "Ask Abraham anything..."
    )
    with st.container(height=400):
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_question:
            with st.chat_message("user"):
                st.markdown(user_question)
            st.session_state["messages"].append(
                {"role": "user", "content": user_question}
                )

            with st.chat_message("assistant"):
                with st.spinner("Reading Abraham's mind..."):
                    start = time.time()
                    response = user_input(
                        user_question,
                        st.session_state["chat_history"]
                    )
                    resp_time = time.time() - start
                    st.markdown(response)
            st.session_state["messages"].append(
                {"role": "assistant", "content": response}
            )
            st.session_state["chat_history"].append(
                (user_question, response)
            )
            st.success(f"Response generated in {resp_time:.2f} seconds")


if __name__ == "__main__":
    main()
