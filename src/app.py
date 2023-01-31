import pandas as pd
import streamlit as st

from fsdl_qa import FSDLQAChain


@st.cache
def get_qa_chain():
    return FSDLQAChain()


if __name__ == "__main__":
    question = st.text_input("QUESTION", "")
    qa_chain = get_qa_chain()

    if question:
        # Execute question against paragraph
        if question != "":
            st.write(question)
            out = qa_chain.query(question=question)
            st.write(out)

        df = pd.DataFrame({
            'feedback column': ["", "good", "bad"]
        })

        option = st.selectbox(
            'Please provide a feedback for the answer:',
            df['feedback column'])

        'You selected: ', option
