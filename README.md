# LLM-QA

QA project using LLM

pip install langchain
pip install openai
pip install srt
pip install faiss-cpu

### start gantry local dev env

cd gantry
tilt up

### start streamlit server

Before running please export the API keys using environment variables:

cd src
export OPENAI_API_KEY=...
export GANTRY_API_KEY=...
export GANTRY_LOGS_LOCATION="http://localhost:5001"
export GANTRY_BYPASS_FIREHOSE="true"

streamlit run app.py

Pipeline data can be found in src/logs each execution is a json file

Next step: log to local stack
