FROM python:3.8.12-buster
COPY vape_api /vape_api
COPY vape_model /vape_model
COPY requirements.txt /requirements.txt
COPY registry_for_api /registry_for_api

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install libgl1 -y

CMD uvicorn vape_api.fast_api:app --host 0.0.0.0
