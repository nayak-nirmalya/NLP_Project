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

# Describes Data, How Many Rows and Columns, Unique Classes
Data_Frame.describe()

# To Display First 'n' Numbers of Rows.
Data_Frame.head(12)

# Prints the Message Content of 'n-th' Row
print(Data_Frame['msg_body'][4], sep = '\n')

# Prints Label
print(Data_Frame.label, sep = '\n')

# As We Need Numerical Value for Training the Model, This Map Method Replaces 'ham' as '0' and 'spam' as '1' in the Dataset.
Data_Frame['label'] = Data_Frame.label.map({'ham' : 0, 'spam' : 1})


# ## USING NLTK LIBRARY TO PRE-PROCESS TEXT DATA 

nltk.download('stopwords')                                # Downlaod 'STOPWORDS'
from nltk.corpus import stopwords                         # Import 'STOPWORDS' from NLTK
from nltk.stem.porter import PorterStemmer                # Import 'PortStemmer' from NLTK

# Making Constructor of PortStemmer Class
ps = PorterStemmer()

print(stopwords.words("english"))                         # Printing StopWords in English

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


print(*review_simplified, len(review_simplified), sep = '\n')


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

# Printing Features
print(len(f_name), f_name, sep = '\n')


# Shape of Sparse Matrix
X.shape


print(*X[0])


# DEVIDING DATA SET INTO TRAINING AND TESTING SETS

from sklearn.model_selection import train_test_split

# Splitting Data to Training and Testing Set with 80 - 20 Split.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.80, shuffle = True, random_state = 0)


print("No. of Rows Being Used for Training Purpose: ", X_train.shape[0])
print("No. of Rows Being Used for Testing Purpose: ", X_test.shape[0])


# MAKING CLASSIFIER MODEL AND TRAINING IT

# Importing Classifier from 'sklearn' Library
from sklearn.naive_bayes import MultinomialNB

# Here We are Using Multinomial Naive Bayse Model for Classification
classifier = MultinomialNB()

# Fitting Training Set to Model with Default Parameters
classifier.fit(X_train, y_train)


# ## PREDICTING ON TEST DATA SET

# Predicted Result Saved to Variable 'y_pred'
y_pred = classifier.predict(X_test)
print(*y_pred)


# PREDICTING ON USER INPUT DATA

# Asks User to Input A Sentece to Check Whether it's a Spam or not and Saves it to a List.
random_test_msg = [ input("Enter Message: ") ]

"""
User Input Message Also Has to be Pre-Processed in Same format as the Testing Data
in Order to Find the Desired Output
"""

review_simplified_test = list()
for _ in random_test_msg:
    review = re.sub('[^a-zA-Z]', ' ', _)
    review = review.lower()
    review = review.split()
    review = [ps.stem(i) for i in review if i not in set(stopwords.words("english"))]
    review = " ".join(review)
    review_simplified_test.append(review)

review_simplified_test

# Making Bag of Word / Sparse Matrix Model for the User Input Message.
X_myDemo = [review_simplified_test[0].split().count(_) for _ in f_name]
arr = np.array(X_myDemo)
print(*arr)

y_myPred = classifier.predict([arr])


y_myPred[0]                                       # 0 - Not-Spam, 1 - Spam
print("Message Entered by User Is ", ["Not ", ""][y_myPred[0]], 'a Spam Message.', sep = '')


# EVALUATING MODEL PERFORMANCE

from sklearn.metrics import confusion_matrix

# Confussion Matrix Gives no. of 'True Positive', 'True Negative', 'False Positive', 'False Negative'
cm = confusion_matrix(y_test, y_pred)
print(cm)

import seaborn as sns

# Ploting Confusion Matrix in a Figurative Manner.
plt.figure(figsize = (4, 4))
sns.heatmap(cm, annot = True);

from sklearn.metrics import classification_report

# Finding Model's Accuracy Score
print(classification_report(y_test, y_pred))