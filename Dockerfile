FROM python:3.8.12-buster
COPY vape_api /vape_api
COPY vape_model /vape_model
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn vape_api.fast:app --host 0.0.0.0
