# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 14:28:08 2022

@author: hp
"""
'''
import nltk

from nltk.stem import SnowballStemmer

Italian_stemmer = SnowballStemmer("italian")

print(Italian_stemmer.stem("Ciao"))
'''

'''
import nltk

from nltk.stem import RegexpStemmer

Reg_stemmer = RegexpStemmer("ing")

print(Reg_stemmer.stem('ingwatch'))
'''

'''
import nltk

from nltk.stem import LancasterStemmer

Lanc_stemmer = LancasterStemmer()

print(Lanc_stemmer.stem('watches'))
'''

'''
import nltk

from nltk.stem import PorterStemmer

word_stemmer = PorterStemmer()

print(word_stemmer.stem('watching'))
'''

'''
import nltk

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize('watches'))
'''

'''
corpus = ['SOME GIBBIRISH death note light yagami, This is the first sentence in our corpus followed by one more sentence to demonstrate Bag of words', 'This is the second sentence in our corpus with a FEW UPPER CASE WORDS and Few Title Case Words']

vocab = []	# empty list for vocabulary

total_words = 0	# to count total words in corpus

for doc in corpus: # iterating through documents in corpus

    token_temp = doc.split() # create tokens

    total_words = total_words + len(token_temp)

for i in range(len(token_temp)):

    if token_temp[i] not in vocab: # to check if word is already in vocab

        vocab.append(token_temp[i])

        vocab.sort()

print(vocab) # Print all the words in vocabulary

print('There are {} words in vocabulary.'.format(len(vocab)))

print('A total of {} words is used in documents.'.format(total_words))
'''

'''
from sklearn.feature_extraction.text import TfidfVectorizer

d0 = 'death note'

d1 = 'light yagami'

d2 = 'kiara'

string = [d0, d1, d2]

tfidf = TfidfVectorizer()

result = tfidf.fit_transform(string)

print('\nidf values:')

for ele1, ele2 in zip(tfidf.get_feature_names(), tfidf.idf_):

    print(ele1, ':', ele2)

print('\nWord indexes:')
print(tfidf.vocabulary_)
print('\ntf-idf value:')
print(result)
print('\ntf-idf values in matrix form:')
print(result.toarray())
'''

'''
import pandas as pd

from nltk.stem import WordNetLemmatizer from nltk.corpus import stopwords

from nltk import pos_tag, word_tokenize

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer from sklearn import svm

from sklearn.metrics import confusion_matrix

data = pd.read_csv("spam.csv", encoding = "latin-1") data = data[['v1', 'v2']]

data = data.rename(columns = {'v1': 'label', 'v2': 'text'}) lemmatizer = WordNetLemmatizer()

stopwords = set(stopwords.words('english'))
 
def review_messages(msg):

    msg = msg.lower()
    return msg

def alternative_review_messages(msg):

    msg = msg.lower()

    nltk_pos = [tag[1] for tag in pos_tag(word_tokenize(msg))] msg = [tag[0] for tag in pos_tag(word_tokenize(msg))]

    wnpos = ['a' if tag[0] == 'J' else tag[0].lower() if tag[0] in ['N', 'R', 'V'] else 'n' for tag in nltk_pos] msg = " ".join([lemmatizer.lemmatize(word, wnpos[i]) for i, word in enumerate(msg)])

    msg = [word for word in msg.split() if word not in stopwords] 
    return msg

data['text'] = data['text'].apply(review_messages)

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size = 0.1, random_state = 1)

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)

svm = svm.SVC(C=1000)

svm.fit(X_train, y_train)

X_test = vectorizer.transform(X_test)

y_pred = svm.predict(X_test)

print(confusion_matrix(y_test, y_pred))

def pred(msg):

    msg = vectorizer.transform([msg])

    prediction = svm.predict(msg)

    return prediction[0]

'''
