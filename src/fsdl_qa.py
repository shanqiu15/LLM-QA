import os
from pathlib import Path
import shutil
import random
import json
import string

import srt

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import VectorDBQAWithSourcesChain
from langchain import OpenAI

DOCS_FOLDER = Path("docs")
TEMPERATURE = 0


class FSDLQAChain:
    def __init__(self, api_key, doc_search=None):
        lecture_text_list, lecture_metadata = parse_lectures()
        srt_text_list, srt_metadata = parse_srt()
        all_text_splits = lecture_text_list + srt_text_list
        all_text_metadata = lecture_metadata + srt_metadata
        self.embeddings = HuggingFaceEmbeddings()
        if not doc_search:
            self.docsearch = FAISS.from_texts(
                all_text_splits, self.embeddings, all_text_metadata)
        else:
            self.docsearch = doc_search
        self.openai_llm = OpenAI(
            temperature=TEMPERATURE, openai_api_key=api_key)
        self.chain = VectorDBQAWithSourcesChain.from_chain_type(
            self.openai_llm, chain_type="stuff", vectorstore=self.docsearch)

    def query(self, question: str):
        answer = self.chain({"question": question}, return_only_outputs=True)
        docs_and_scores = self.chain.vectorstore.similarity_search_with_score(
            question, k=self.chain.k, **self.chain.search_kwargs
        )
        log_docs_and_score = [{"text": item[0].page_content,
                               "source": item[0].metadata["source"], "score": item[1].item()} for item in docs_and_scores]
        docs = [doc for doc, _ in docs_and_scores]
        prompt_input = self.chain.combine_documents_chain._get_inputs(
            docs, **{"question": question})
        final_prompt = self.chain.combine_documents_chain.llm_chain.prompt.format(
            **prompt_input)
        pipeline = json.load(open("pipeline_template.json"))
        for node in pipeline["nodes"]:
            match node["name"]:
                case "Embedding Index":
                    node["data"] = [{"index_id": "local_faiss_v0"}]
                    node["metadata"] = {"model_id": self.embeddings.model_name}
                case "Question":
                    node["data"] = [{"question": question}]
                case "Generate Embedding":
                    node["metadata"] = {"model_id": self.embeddings.model_name}
                case "Question Embedding":
                    node["data"] = [
                        {"embedding": self.chain.vectorstore.embedding_function(question)}],
                case "Prompt Template":
                    node["data"] = [
                        {"prompt template": self.chain.combine_documents_chain.llm_chain.prompt.template}],
                case "KNN":
                    node["metadata"] = {"K": self.chain.k,
                                        "Env": "FAISS"}
                case "Top N Neighbors":
                    node["data"] = log_docs_and_score
                case "Prompt":
                    node["data"] = [{"Prompt": final_prompt}]
                case "LLM":
                    node["metadata"] = {
                        "model": self.openai_llm.model_name,  "temperature": TEMPERATURE}
                case "Answer":
                    node["data"] = [{"answer": answer}]
        file_name = f"logs/log_session_{random.randint(0, 10000)}.json"
        with open(file_name, "w") as outfile:
            json.dump(pipeline, outfile, indent=4)

        return answer, file_name


def get_lecture_titles():
    return {
        1: "lecture-1-course-vision-and-when-to-use-ml",
        2: "lecture-2-development-infrastructure-and-tooling",
        3: "lecture-3-troubleshooting-and-testing",
        4: "lecture-4-data-management",
        5: "lecture-5-deployment",
        6: "lecture-6-continual-learning",
        7: "lecture-7-foundation-models",
        8: "lecture-8-teams-and-pm",
        9: "lecture-9-ethics"
    }


def get_srt_urls():
    return {
        1: "https://www.youtube.com/watch?v=-Iob-FW5jVM",
        2: "https://www.youtube.com/watch?v=BPYOsDCZbno",
        3: "https://www.youtube.com/watch?v=RLemHNAO5Lw",
        4: "https://www.youtube.com/watch?v=Jlm4oqW41vY",
        5: "https://www.youtube.com/watch?v=W3hKjXg7fXM",
        6: "https://www.youtube.com/watch?v=nra0Tt3a-Oc",
        7: "https://www.youtube.com/watch?v=Rm11UeGwGgk",
        8: "https://www.youtube.com/watch?v=a54xH6nT4Sw",
        9: "https://www.youtube.com/watch?v=7FQpbYTqjAA"
    }


def parse_lectures():
    lecture_md_filenames = [
        elem for elem in DOCS_FOLDER.iterdir() if elem.is_file() and "lecture" in str(elem) and str(elem).endswith("md")]
    lecture_titles = get_lecture_titles()

    lecture_texts = {}
    for fn in lecture_md_filenames:
        idx = int("".join(elem for elem in str(
            fn) if elem in string.digits))
        lecture = fn.open().read()
        lecture_texts[idx] = lecture

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    lecture_texts_split = {idx: text_splitter.split_text(
        lecture_text) for idx, lecture_text in lecture_texts.items()}
    website_url_base = "https://fullstackdeeplearning.com/course/2022/"
    source_urls = {idx: website_url_base +
                   title for idx, title in lecture_titles.items()}

    # Source URL as the key, lecture split list as the value
    url_to_text_split = dict([(url, text_split) for url, text_split in zip(
        source_urls.values(), lecture_texts_split.values())])
    return get_flat_text_metadata(url_to_text_split)


def timestamp_from_timedelta(timedelta):
    return int(timedelta.total_seconds())


def create_srt_texts_and_metadatas(subtitles, base_url):
    query_params_format = "&t={start}s"
    texts, metadatas = [], []

    for subtitle in subtitles:
        raw_text = subtitle.content
        text = subtitle.content.strip()
        start = timestamp_from_timedelta(subtitle.start)
        url = base_url + query_params_format.format(start=start)

        texts.append(text)
        metadatas.append(url)

    return texts, metadatas


def parse_srt():
    url_to_text_split = {}
    srt_filenames = [
        elem for elem in DOCS_FOLDER.iterdir() if elem.is_file() and str(elem).endswith("srt")]
    srt_urls = get_srt_urls()

    for fn in srt_filenames:
        idx = int("".join(elem for elem in str(fn) if elem in string.digits))
        srt_url = srt_urls[idx]

        srt_text = fn.open().read()
        subtitles = list(srt.parse(srt_text))
        texts, metadatas = create_srt_texts_and_metadatas(subtitles, srt_url)

        for text, url in zip(texts, metadatas):
            url_to_text_split[url] = [text]
    return get_flat_text_metadata(url_to_text_split)


def get_flat_text_metadata(url_to_text_split):
    all_text_splits = []
    all_text_metadata = []

    for source_url, text_splits in url_to_text_split.items():
        for text in text_splits:
            all_text_splits.append(text)
            all_text_metadata.append({"source": source_url})
    return all_text_splits, all_text_metadata
