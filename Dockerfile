FROM python:3.8.12-buster
COPY brainsignals brainsignals
COPY .registry .registry
COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN pip install .
COPY .env .env
RUN python -c 'from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv())'
CMD uvicorn brainsignals.api:app --reload
