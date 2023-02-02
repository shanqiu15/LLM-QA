import faiss
import pandas as pd
import streamlit as st
import gantry
import os
import json
import base64
import hashlib
import boto3
import numpy as np
import gantry.dataset as gdataset
import gantry
from pathlib import Path
from fsdl_qa import FSDLQAChain
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document

DATASET_NAME = "fsdl_document_index"
APP_NAME = f"llm_qa_test_{DATASET_NAME}"


@st.cache
def get_qa_chain():
    gantry.init()
    workspace = Path("datasets").resolve()
    gdataset.set_working_directory(str(workspace))
    embedding_file = workspace/DATASET_NAME/"artifacts"/"embeddings.json"
    embedding_data = json.load(embedding_file.open())

    docstore_dict = {}
    index_to_id = {}
    text_embeddings = []
    for idx, item in enumerate(embedding_data):
        text_embeddings.append(item['embedding'])
        index_to_id[idx] = item["id"]
        docstore_dict[item["id"]] = Document(
            page_content=item['text_chunks'], metadata=item["metadata"])

    index = faiss.IndexFlatL2(len(text_embeddings[0]))
    index.add(np.array(text_embeddings, dtype=np.float32))

    docstore = InMemoryDocstore(docstore_dict)
    doc_search = FAISS(HuggingFaceEmbeddings().embed_query, index=index,
                       docstore=docstore, index_to_docstore_id=index_to_id)

    return FSDLQAChain(os.getenv('OPENAI_API_KEY'), doc_search=doc_search)


if __name__ == "__main__":
    question = st.text_input("QUESTION", "")
    qa_chain = get_qa_chain()

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
