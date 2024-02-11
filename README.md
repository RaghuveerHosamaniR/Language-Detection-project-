# Language-Detection-project-
#Language Detection with Machine Learning
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
print(os.getcwd())

import os

# Specify the new directory path with quotes
new_directory = r'G:\Kalasalingam college pdf Notes\Machine Learning\Project ML\Launguages'

# Change the current working directory
os.chdir(new_directory)

# Verify the change by printing the current working directory
print("Current working directory:", os.getcwd())

data = pd.read_csv("Language Detection.csv")
print(data.head())

data.isnull().sum()

data["Language"].value_counts()

x = np.array(data["Text"])
y = np.array(data["Language"])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.33, 
                                                    random_state=42)

model = MultinomialNB()
model.fit(X_train,y_train)
model.score(X_test,y_test)

#Now let’s use this model to detect the language of a text by taking a user input:
user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)
