from sklearn.datasets import fetch_20newsgroups
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pickle


categories = ['talk.politics.guns', 'sci.crypt', 'alt.atheism', 'comp.graphics', 'rec.autos']
data = fetch_20newsgroups(subset='all',
                          categories=categories,
                          shuffle=False,
                          remove=('headers', 'footers', 'quotes'))

text, label = data.data, data.target

df = pd.DataFrame({'text':text, 'label':label})

#clear
stemmer = SnowballStemmer('English')
def stem(text):
    word_list = word_tokenize(text)
    lemmatized_doc = ""
    for text in word_list:
        lemmatized_doc = lemmatized_doc + " " + stemmer.stem(text)
    return lemmatized_doc.strip()

def rmv_special_char(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)

def rmv_stopwords(text):
    stop_w = stopwords.words('english')
    text = [word for word in text.split() if word not in stop_w]
    text = ' '.join(x.lower() for x in text)
    return text

df['text'] = df['text'].apply(stem).apply(rmv_special_char).apply(rmv_stopwords)

count_V = CountVectorizer()
bow = count_V.fit_transform(df['text'])
with open('bow.pkl', 'wb') as f:
    pickle.dump(bow, f)

tfidf_V = TfidfVectorizer()
tdidf = tfidf_V.fit_transform(df['text'])
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tdidf, f)
