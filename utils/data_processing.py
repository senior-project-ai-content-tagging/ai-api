import re
from cleantext import clean
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize

def clean_data(text):
    text = text.lower()
    text = clean(text, no_emoji = True, no_line_breaks = True)
    text = re.sub('[^\w\s]+', '', text)
    text = re.sub('\d+', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

def lemmatize_word(list_word):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in list_word]

def remove_stopword(tokens):
    stop_words = set(stopwords.words("english"))
    token = [w for w in tokens if not w in stop_words]
    return token

def preprocessing(text):
    text = clean_data(text)
    # tokenize 
    token = word_tokenize(text)
    # remove stop words from tokens
    stop_token = remove_stopword(token)
    # lemmatise tokens
    lemma_word = lemmatize_word(stop_token)
    text = ' '.join(word for word in lemma_word)
    return text
