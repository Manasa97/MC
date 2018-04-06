import csv
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

train = pd.read_csv("train.csv")
test = pd.read_csv("qaps_2000_cleaned.csv")

train.columns = ["coarse", "fine", "text"]
test.columns = ["text", "coarse", "fine"]

pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer()),
    ('tfidf',              TfidfTransformer()),
    ('classifier',         OneVsRestClassifier(LinearSVC()))
])

test_example = ['Who killed Abraham Lincoln?']


lb = LabelBinarizer()
z = lb.fit_transform(train["coarse"])

pipeline.fit(train["text"], z)
predicted = pipeline.predict(test["text"])
predicted_example = pipeline.predict(test_example)
predictions = lb.inverse_transform(predicted)
predictions_example = lb.inverse_transform(predicted_example)
print(len(predictions))


accuracy = accuracy_score(test["coarse"], predictions)
print(accuracy)

print('Sample question: ', test_example)
print('Class: ', predictions_example)

'''
LinearSVC (accuracy 0.70) - fastest
svm.SVC linear kernel (accuracy 0.67)
'''
