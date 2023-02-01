import pandas as pd
import streamlit as st
import gantry
import os
import json
import base64
import hashlib

from fsdl_qa import FSDLQAChain

APP_NAME = "llm_qa_test_2_1"
os.environ["GANTRY_LOGS_LOCATION"] = "https://app.staging.gantry.io"


@st.cache
def get_qa_chain():
    return FSDLQAChain(os.getenv('OPENAI_API_KEY'))


if __name__ == "__main__":
    question = st.text_input("QUESTION", "")
    qa_chain = get_qa_chain()

    gantry.init()

    if question:
        with st.form("QA_FORM"):
            st.write(question)
            out, pipeline_file = qa_chain.query(question=question)
            st.write(out)

            df = pd.DataFrame({
                'feedback column': ["", "good", "bad"]
            })

            option = st.selectbox(
                'Please provide a feedback for the answer:',
                df['feedback column'])

            if option:
                gantry.log_records(
                    APP_NAME,
                    inputs=[{"question": question,
                             "pipeline": f"s3://bucket/{pipeline_file}"}],
                    outputs=[{"output": json.dumps(out)}],
                    feedbacks=[{"customer_feedback": option}],
                    join_keys=[base64.b64encode(hashlib.sha256(
                        question.encode('utf-8')).digest()).decode("utf-8")],
                    tags={"env": "llm-demo"}
                )

            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit")
            if submitted:
                st.write(
                    f"You feedback to the generated answer has been submitted. Feedback: {option}")
