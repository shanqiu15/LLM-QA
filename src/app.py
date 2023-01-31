import pandas as pd
import streamlit as st

from fsdl_qa import FSDLQAChain


if __name__ == "__main__":
    question = st.text_input("QUESTION", "")
    qa_chain = FSDLQAChain()

    if question:
        # Execute question against paragraph
        if question != "":
            out = qa_chain.query(question=question)
            st.write(out)

        df = pd.DataFrame({
            'feedback column': ["", "good", "bad"]
        })

        option = st.selectbox(
            'Please provide a feedback for the answer:',
            df['feedback column'])

        'You selected: ', option
