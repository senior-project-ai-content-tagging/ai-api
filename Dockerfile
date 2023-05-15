FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y libpq-dev build-essential default-libmysqlclient-dev libpq5


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

COPY . /app/

EXPOSE 8080

CMD ["python", "server.py"]
