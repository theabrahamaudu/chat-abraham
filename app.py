import streamlit as st
from src.pdf_parser import (
    get_vector_store,
    user_input,
    get_pdf_text,
    get_text_chunks
)


def main():
    st.set_page_config(page_title="ChatAbraham")
    st.header("ChatAbraham")
    
    if st.button("Configure App"):
        with st.spinner("Configuring App..."):
            st.session_state["conversation_history"] = None
            pdf_text = get_pdf_text("./data/UNFINISHED_Audu_2020.pdf")
            text_chunks = get_text_chunks(pdf_text)
            get_vector_store(text_chunks)
            st.success("App Configured!")

    user_question = st.chat_input(
        "Ask Abraham anything..."
    )

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
            
        with st.chat_message("assistant"):
            response = user_input(user_question)
            st.write(response)


if __name__ == "__main__":
    main()