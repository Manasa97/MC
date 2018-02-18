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


lb = LabelBinarizer()
z = lb.fit_transform(train["coarse"])

pipeline.fit(train["text"], z)
predicted = pipeline.predict(test["text"])
predictions = lb.inverse_transform(predicted)
print(len(predictions))


accuracy = accuracy_score(test["coarse"], predictions)
print(accuracy)

