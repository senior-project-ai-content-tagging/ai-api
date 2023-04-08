# Step 1: Load libraries
import pickle

from nltk.jsontags import json
from utils.data_processing import preprocessing
import joblib
from google.cloud import pubsub_v1
import psycopg2
import os

PREPARED_CONTENT_SUBSCRIPTION = 'projects/senior-project-364818/subscriptions/content-preparing.prepared-content-sub'

FIND_ID_AND_CONTENT_EN_BY_TICKET_ID_SQL = """
SELECT contents.id, content_en FROM contents
JOIN tickets ON contents.id = tickets.content_id
WHERE tickets.id = %s
"""

FIND_CATEGORY_ID_BY_NAME_SQL = """
SELECT id from categories
WHERE name = %s
"""

CREATE_CONTENT_CATEGORY_SQL = """
INSERT INTO contents_categories (id, content_id, category_id)
VALUES (nextval('contents_categories_id_seq'), %s, %s)
"""

UPDATE_TICKET_STATUS_SQL = """
UPDATE tickets SET status = %s
WHERE id = %s
"""

conn = psycopg2.connect(database=os.environ["DB_NAME"],
                        host=os.environ["DB_HOST"],
                        user=os.environ["DB_USER"],
                        password=os.environ["DB_PASS"],
                        port=os.environ["DB_PORT"])

# Step 2: Load saved model
# with open('knn_eng_tune.sav', 'rb') as f:
# my_model = pickle.load(open('./knn_eng_tune.sav', 'rb'))
# my_model = load_model('./model_lstm_eng.h5')
my_model = joblib.load('./random_forest_eng_tune.joblib')

tfidf_eng, train_data = pickle.load(open('./tfidf_eng.pkl', 'rb'))


# tokenizer = Tokenizer(num_words=10000)
# seq = tokenizer.texts_to_sequences(preprocessing_data)
# pad_seq = pad_sequences(seq, maxlen=300)

def callback(incoming_message):
    data = incoming_message.data.decode('utf-8')
    json_message = json.loads(data)
    ticket_id = json_message["ticketId"]

    cursor = conn.cursor()
    cursor.execute(FIND_ID_AND_CONTENT_EN_BY_TICKET_ID_SQL, (str(ticket_id), ))
    (content_id, content_en) = cursor.fetchone()
    preprocessing_data = preprocessing(content_en)
    print(preprocessing_data)
    vector_text = tfidf_eng.transform([preprocessing_data])
    result = my_model.predict(vector_text)
    for category in result:
        print(category)
        cursor.execute(FIND_CATEGORY_ID_BY_NAME_SQL, (category, ))
        category_id = cursor.fetchone()[0]
        cursor.execute(CREATE_CONTENT_CATEGORY_SQL, (str(content_id), str(category_id)))
        cursor.execute(UPDATE_TICKET_STATUS_SQL, ("DONE", str(ticket_id)))

    conn.commit()
    incoming_message.ack()

with pubsub_v1.SubscriberClient() as subscriber:
    future = subscriber.subscribe(PREPARED_CONTENT_SUBSCRIPTION, callback)
    try:
        future.result()
    except KeyboardInterrupt:
        future.cancel()
