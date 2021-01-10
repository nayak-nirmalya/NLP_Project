import flask
from flask import request, jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = True

#!/usr/bin/env python
# coding: utf-8

# SPAM E-MAIL DETECTION USING NATURAL LANGUAGE PROCESSING

# IMPORTING REQUIRED LIBRARIES

import pandas as pd                        # For Making Data Frame From Dataset
import numpy as np                         # For Making n-Dimensional Array
import matplotlib.pyplot as plt            # Library for Visualising Data
import re                                  # Regular Expression Library
import nltk                                # Natural Language Toolkit Library for Text Based Data


# DATA PRE-PROCESSING

# IMPORTING AND UNDERSTANDIG DATASET


# Here "pandas" Library is Used to Save the Source Data in a Data Frame Separated by Tab.
# Label Decides Wheather a Message or E-Mail is Spam or Not 
# "msg_body" Describes the Content of Message
Data_Frame = pd.read_table("smsspamcollection/SMSSpamCollection", sep='\t', names = ['label', 'msg_body'])


# As We Need Numerical Value for Training the Model, This Map Method Replaces 'ham' as '0' and 'spam' as '1' in the Dataset.
Data_Frame['label'] = Data_Frame.label.map({'ham' : 0, 'spam' : 1})


# ## USING NLTK LIBRARY TO PRE-PROCESS TEXT DATA 

nltk.download('stopwords')                                # Downlaod 'STOPWORDS'
from nltk.corpus import stopwords                         # Import 'STOPWORDS' from NLTK
from nltk.stem.porter import PorterStemmer                # Import 'PortStemmer' from NLTK

# Making Constructor of PortStemmer Class
ps = PorterStemmer()

# Creating an Empty List
review_simplified = list() 

# Looping Through All Elements of DataFrame
for _ in Data_Frame['msg_body']:
    
    # Removing Every Thing Except 'a - z' & 'A - Z' and Replacing It With "<SPACE>"
    review = re.sub('[^a-zA-Z]', ' ', _) 
    
    # Making All The Words to Lower Case
    review = review.lower()
    
    # Splitting Sentence into List of Words
    review = review.split()
    
    # Python List Comprehnsion Method to Remove Stopwords from List and Stemming the Remaining Word
    review = [ps.stem(i) for i in review if i not in set(stopwords.words("english"))]
    
    # Joining the Words in a List Back to Sentence
    review = " ".join(review)
    
    # Appending the Processed Sentence to "review_simplified" List
    review_simplified.append(review)


# ## MAKING BAG OF WORD MODEL

from sklearn.feature_extraction.text import CountVectorizer

# Making Constructor of CountVectorizer Class
cv = CountVectorizer()

# Fitting Review_Simplified and Transforming it into a Sparse Matrix
X = cv.fit_transform(review_simplified).toarray()

# Saving Unique Feature Names to a Variable for Future Use
f_name = cv.get_feature_names()

# Saving Labels of Data_Frame to "y" Variable
y = Data_Frame['label'].values


# DEVIDING DATA SET INTO TRAINING AND TESTING SETS

from sklearn.model_selection import train_test_split

# Splitting Data to Training and Testing Set with 80 - 20 Split.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.80, shuffle = True, random_state = 0)

# MAKING CLASSIFIER MODEL AND TRAINING IT

# Importing Classifier from 'sklearn' Library
from sklearn.naive_bayes import MultinomialNB

# Here We are Using Multinomial Naive Bayse Model for Classification
classifier = MultinomialNB()

# Fitting Training Set to Model with Default Parameters
classifier.fit(X_train, y_train)


@app.route('/', methods=['GET'])
def home():
    return jsonify({'Welome': "API to Access NLP Model for Spam Detection."})


@app.route('/api/v1/get_result', methods=['POST', 'GET'])
def get_predicted_result():
	if request.method == 'POST':
		req_data = request.get_json()
		usr_msg = req_data["msg_content"]
		processed_msg = cv.transform([usr_msg])
		usr_pred = classifier.predict(processed_msg)
		if (usr_pred[0]) == 1:
			return jsonify({'result': "Spam"})
		else:
			return jsonify({'result': "Ham"})

app.run()
