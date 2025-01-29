import ollama
from streamlit_pdf_viewer import pdf_viewer

import streamlit as st
from llm import PDFChatbot

st.set_page_config(layout="wide")


def get_models_list():
    models = ollama.list()
    models = [m.model for m in models.models]
    return [None, *models]


def initialize_session_state():
    if "pdf_is_loaded" not in st.session_state:
        st.session_state['pdf_is_loaded'] = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdf_chatbot" not in st.session_state:
        st.session_state.pdf_chatbot = None
    if "model_selected" not in st.session_state:
        st.session_state.model_selected = None


def main():
    st.title("Chat with your PDF!")

    st.markdown("""
    <style>
    .stForm {
        position: fixed;
        bottom: 10px;
        left: 350px;
        right: 0;
        width: 100%;
        max-width: 800px;
        margin: 0 auto;
        padding: 10px;
        z-index: 1000;
    }
    .main .block-container {
        padding-bottom: 100px;
        max-width: 800px;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        model_name = st.selectbox(
            "Select a model", get_models_list()
        )
        pdf = st.file_uploader("Upload your PDF", type=["pdf"])
        if pdf is not None and not st.session_state['pdf_is_loaded']:
            st.session_state.model_selected = model_name
            print('model_selected', st.session_state.model_selected)
            st.session_state.pdf_bytes = pdf.getvalue()
            st.session_state["pdf_chatbot"] = PDFChatbot()
            st.session_state.pdf_chatbot.process_pdf(pdf)
            st.write("PDF uploaded successfully!")
            st.session_state['pdf_is_loaded'] = True
            st.session_state['pdf'] = pdf
            pdf_viewer(input=st.session_state.pdf_bytes)

    if st.session_state['pdf_is_loaded']:
        chat_container = st.container(height=500)
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.write(f"You: {message['content']}")
                else:
                    st.write(f"Assistant: {message['content']}")

        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([6, 1])
            with col1:
                user_input = st.text_input(
                    "Message:", key="user_input", label_visibility="collapsed")
            with col2:
                submit_button = st.form_submit_button("Send")

        if submit_button and user_input:
            st.session_state.messages.append(
                {"role": "user", "content": user_input})
            response = st.session_state.pdf_chatbot.query_pdf(user_input)
            response = response.replace(
                "<think>", "\n```\n").replace("</think>", "```")
            st.session_state.messages.append(
                {"role": "assistant", "content": response})


if __name__ == "__main__":
    initialize_session_state()
    main()
