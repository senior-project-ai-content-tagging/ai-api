FROM python:3.8-slim-buster

ENV MODEL_URL="https://storage.googleapis.com/ai-modal/ai-modal.joblib"
ENV TFIDF_URL="https://drive.google.com/uc?export=download&id=1FN63hj5A_Y28TUBgZ6nn8oo--_85ywE7"

RUN apt-get update && apt-get install -y libpq-dev build-essential default-libmysqlclient-dev libpq5 wget


WORKDIR /app

COPY Pipfile Pipfile.lock /app/

# Install dependencies
RUN pip install pipenv 
RUN pipenv install --system --deploy --ignore-pipfile

# setup nltk
RUN pip install nltk
ENV NLTK_DATA=/usr/share/nltk_data
RUN mkdir -p $NLTK_DATA
COPY download_nltk_data.py .
RUN python download_nltk_data.py

# donwload model (large file)
RUN pip install gdown
RUN gdown "$MODEL_URL" -O model.joblib
RUN gdown "$TFIDF_URL" -O tfidf.pkl

COPY . /app/

EXPOSE 8080

CMD ["python", "server.py"]
