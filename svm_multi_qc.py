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
import pickle

train = pd.read_csv("train.csv")
test = pd.read_csv("qaps_2000_cleaned.csv")

train.columns = ["coarse", "fine", "text"]
test.columns = ["text", "coarse", "fine"]

pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer()),
    ('tfidf',              TfidfTransformer()),
    ('classifier',         OneVsRestClassifier(LinearSVC()))
])
'''
CountVectorizer() - gets the vocabulary
TfidfTransformer converts words to frequencies

'''
test_example = ['how many apples?']

#Binarise labels in a one vs all fashion
lb = LabelBinarizer()
#transform binary targets to a column vector eg.([yes,no,no]) -> [1,0,0]
z = lb.fit_transform(train["coarse"])

pipeline.fit(train["text"], z)

pickled_file = "SVM_pipeline.txt"
pickle.dump(pipeline, open(pickled_file,'wb'))
pickle.dump(lb, open("lb_pickled_file.txt",'wb'))

pipeline1 = pickle.load(open(pickled_file,'rb'))
predicted = pipeline1.predict(test["text"])
predicted_example = pipeline1.predict(test_example)

#Turn labels back into original multi-class
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
