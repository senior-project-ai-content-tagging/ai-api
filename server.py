import base64
import pickle
import os
import json
import psycopg2
import joblib
from flask import Flask, request
from utils.data_processing import preprocessing


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

conn_str = f'host={os.environ.get("DB_HOST")} user={os.environ.get("DB_USER")} password={os.environ.get("DB_PASS")} dbname={os.environ.get("DB_NAME")} options=project=delicate-hat-661219 sslmode=require'
conn = psycopg2.connect(conn_str)

# Step 2: Load saved model
# with open('knn_eng_tune.sav', 'rb') as f:
# my_model = pickle.load(open('./knn_eng_tune.sav', 'rb'))
# my_model = load_model('./model_lstm_eng.h5')
my_model = joblib.load('./random_forest_eng_tune.joblib')

tfidf_eng, train_data = pickle.load(open('./tfidf_eng.pkl', 'rb'))

app = Flask(__name__)


def callback(json_message):
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
    cursor.close()


@app.route("/predict", methods=["POST"])
def predict():
    envelope = request.get_json()
    if not envelope:
        msg = "no Pub/Sub message revice"
        print(f"error: {msg}")
        return f"Bad Request: {msg}", 400
    if not isinstance(envelope, dict) or "message" not in envelope:
        msg = "invalid Pub/Sub message format"
        print(f"error: {msg}")
        return f"Bad Request: {msg}", 400

    pubsub_message = envelope["message"]

    if isinstance(pubsub_message, dict) and "data" in pubsub_message:
        data = base64.b64decode(pubsub_message["data"]).decode("utf-8").strip()
        try:
            json_message = json.loads(data)
            callback(json_message)
        except json.JSONDecodeError:
            print()

    print(f"Hello {data}!")

    return ("", 204)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
