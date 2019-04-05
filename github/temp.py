import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split






message_data = pd.read_csv("spam.csv",encoding = "latin")
message_data.head()

message_data=message_data.rename(columns = {'v1':'Spam/Not_Spam','v2':'message'})

message_data_copy=message_data['message'].copy()



import re, string, unicodedata
import nltk

import contractions

import inflect

from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

























def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text






def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)






















def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words






def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(stems)
    return  lemmas





def call_all(sample):
    
    sample = denoise_text(sample)
   

    
    sample = replace_contractions(sample)
    

    
    
    words = nltk.word_tokenize(sample)
   

    
    
    
    words = normalize(words)

    
    
    
    
    
    lemmas = stem_and_lemmatize(words)
    str1=" ".join(lemmas)
    return str1
    

    
    
    
    
for i , line in enumerate(message_data_copy):
    message_data_copy[i]=call_all(line)




vectorizer = TfidfVectorizer("english")


message_mat = vectorizer.fit_transform(message_data_copy)





message_train, message_test, spam_nospam_train, spam_nospam_test = train_test_split(message_mat, 
                                                        message_data['Spam/Not_Spam'], test_size=0.3, random_state=20)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(message_train, spam_nospam_train)
pred = Spam_model.predict(message_test)




print("Results of test set")


accuracy_score(spam_nospam_test,pred)








from sklearn import metrics

target_names=['ham','spam']

print(metrics.classification_report(spam_nospam_test,pred,target_names=target_names))




array = metrics.confusion_matrix(spam_nospam_test, pred) 
import seaborn as sn 
import pandas as pd 

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline') 

df_cm = pd.DataFrame(array, ["spam","ham"], 
["Spam","Ham"]) 

sn.set(font_scale=1.4)#for label size 
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size







