FROM python:3.12-slim-bookworm

WORKDIR /project

RUN apt-get update && apt-get install gcc -y

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY hiscox_tfidf hiscox_tfidf

EXPOSE 8000

ENTRYPOINT ["python", "-m", "fastapi", "run", "--port", "8080", "./hiscox_tfidf/main.py"]