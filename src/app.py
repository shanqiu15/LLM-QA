import pandas as pd
import streamlit as st
import gantry
import os
import json
import base64
import hashlib
import boto3

from fsdl_qa import FSDLQAChain

APP_NAME = "llm_qa_test_2_1"


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

            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit")
            if submitted:
                s3_client = boto3.client(
                    "s3",
                    endpoint_url="http://localhost:4566",
                )
                response = s3_client.upload_file(
                    pipeline_file, "dev", f"llm/{pipeline_file}")
                st.write(
                    f"Upload pipeline file to bucket: dev, key: llm/{pipeline_file}")

                gantry.log_records(
                    APP_NAME,
                    inputs=[{"question": question,
                            "pipeline": f"s3://dev/llm/{pipeline_file}"}],
                    outputs=[{"output": json.dumps(out)}],
                    feedbacks=[{"customer_feedback": option}],
                    join_keys=[base64.b64encode(hashlib.sha256(
                        question.encode('utf-8')).digest()).decode("utf-8")],
                    tags={"env": "llm-demo"}
                )

                st.write("Log session data to gantry")
