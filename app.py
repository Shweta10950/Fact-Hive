
import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#load data
news_df = pd.read_csv('train.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author']+" "+news_df['title']
X = news_df.drop('label',axis=1)
Y= news_df['label']

# define stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ ps.stem(word) for word in stemmed_content if not word in stopwords.words('english') ]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

#apply stemming function to content column
news_df['content'] = news_df['content'].apply(stemming)
 
# vectorize data
X = news_df['content'].values
Y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,stratify=Y, random_state=1)

#fit logistic Regression model
model = LogisticRegression()
model.fit(X_train,Y_train)


#website
st.title('FactHive')
input_text = st.text_input('Enter news Article')

def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write('The News is Fake')
    else:
        st.write('The News Is Real')