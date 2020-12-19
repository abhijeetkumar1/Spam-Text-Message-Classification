# -*- coding: utf-8 -*-
"""Spam Text Message Classification .ipynb


# 1) Data Preprocessing

## Importing the libraries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""## Importing the data set"""

df = pd.read_csv('/content/spam.tsv',sep='\t')

df.head()

# check null values in dataset
df.isnull().sum()
# no null values in dataset

df.describe()

# check number of ham and spam
df['label'].value_counts()

# 4825 messages are legitimate that is ham
# and 747 messages are spam messages

"""## Balancing the data"""

# select ham data
ham = df[df['label']=='ham']
ham.head()

# select spam data
spam = df[df['label']=='spam']
spam.head()

# check the shape of data
ham.shape, spam.shape

spam.shape[0] # output = no of samples in spam data

# now we have to select 747 samples from ham to balence the data

ham = ham.sample(spam.shape[0])

ham.shape

# check the shape of data
ham.shape, spam.shape

# size of ham and spam data is same, now this is the balenced data

# append spam data into ham data
data = ham.append(spam,ignore_index=True)

data.head()
# at the starting we have all the ham data

data.tail()
# at the end we have all the spam data

data.shape # final shape of the data

"""## Data Visualization"""

# plot histogram of length for ham messages
plt.hist(data[data['label']=='ham']['length'], bins=100, alpha=0.7)
plt.show()
# from the histogram we can say that, the number of charactors in ham messages are less than 100

# plot histogram of length for ham and spam both
plt.hist(data[data['label']=='ham']['length'], bins=100, alpha=0.7)
plt.hist(data[data['label']=='spam']['length'], bins=100, alpha=0.7)
plt.show()

# It looks like there's a small range of values where a message is more likely to be spam than ham

# plot histogram of punct for ham and spam both
plt.hist(data[data['label']=='ham']['punct'], bins=100, alpha=0.7)
plt.hist(data[data['label']=='spam']['punct'], bins=100, alpha=0.7)
plt.show()

# here we are not getting more information

"""## Split the data into train & test sets"""

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data['message'],data['label'],test_size=0.3,
                                                    random_state=0, shuffle=True)

x_train

y_train

"""# 2) Building the Model (Random Forest)"""

from sklearn.pipeline import Pipeline
# there will be lot of repeated processes for training and testing the dataset separately,
# to avoid that we are using pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
# we are importing TfidfVectorizer to utilize bag of words model in sklearn

from sklearn.ensemble import RandomForestClassifier

classifier = Pipeline([('tfidf', TfidfVectorizer()), ('classifier',RandomForestClassifier(n_estimators=100))])

classifier.fit(x_train, y_train)

# all the parameters that you can see while training the model are the default parameters

"""# 3) Predicting the results (Random Forest)"""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = classifier.predict(x_test)

# confusion_matrix
confusion_matrix(y_test, y_pred)

# ham and spam
# spam and ham

# classification_report
print(classification_report(y_test, y_pred))
# we are getting almost 95% accuracy

a1 = accuracy_score(y_test, y_pred)
accuracy_rf = round(a1*100,3)
print(accuracy_rf)

# Predict a real message
classifier.predict(['Hello, You are learning operating system'])

classifier.predict(['Hope you are doing good and learning new things !'])

classifier.predict(['Congratulations, You won a lottery ticket worth $1 Million ! To claim call on 446677'])

classifier.predict(['Amazon is sending you a refund of $32.64. Please reply with your bank account and routing number to receive your refund.'])

"""# 4) Building the model (SVM)"""

from sklearn.svm import SVC

classifier = Pipeline([('tfidf', TfidfVectorizer()), ('classifier',SVC(C=100,gamma='auto'))])

classifier.fit(x_train, y_train)

"""# 5) Predicting the results (SVM)"""

y_pred = classifier.predict(x_test)

# confusion_matrix
confusion_matrix(y_test, y_pred)

a2 = accuracy_score(y_test, y_pred)
accuracy_svm = round(a2*100,3)
print(accuracy_svm)

# Predict a real message
classifier.predict(['Hello, You are learning atural Language Processing'])

classifier.predict(['Hope you are doing good and learning new things !'])

classifier.predict(['Congratulations, You won a lottery ticket worth $1 Million ! To claim call on 446677'])

"""# 6) Building the model (Naive Bayes)"""

from sklearn.naive_bayes import MultinomialNB

classifier = Pipeline([('tfidf', TfidfVectorizer()), ('classifier',MultinomialNB())])

classifier.fit(x_train, y_train)

"""# 7) Predicting the results (Naive Bayes)"""

y_pred = classifier.predict(x_test)

# confusion_matrix
confusion_matrix(y_test, y_pred)

a3 = accuracy_score(y_test, y_pred)
accuracy_nb = round(a3*100,3)
print(accuracy_nb)

# Predict a real message
classifier.predict(['Hello, You are learning atural Language Processing'])

classifier.predict(['Congratulations, You won a lottery ticket worth $1 Million ! To claim call on 446677'])

classifier.predict(['Amazon is sending you a refund of $32.64. Please reply with your bank account and routing number to receive your refund.'])

"""# 8) Accuracy Bar Chart for models (Random Forest, SVM, Naive Bayes)"""

models = ['Random Forest','SVM','Naive Bayes']
accuracylist = [accuracy_rf,accuracy_svm,accuracy_nb]

New_Colors = ['green','blue','purple']
plt.bar(models, accuracylist, color=New_Colors)
plt.title('Accuracy Bar Chart of used models', fontsize=14)
plt.xlabel('Model Name', fontsize=14)
plt.ylabel('Accuracy Percentage', fontsize=14)
plt.grid(True)
plt.show()

models = ['Random Forest','SVM','Naive Bayes']
accuracylist = [accuracy_rf,accuracy_svm,accuracy_nb]
plt.barh(models, accuracylist)

for index, value in enumerate(accuracylist):
    plt.text(value, index, str(value))
