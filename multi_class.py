import csv
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

train_data = np.array(list(csv.reader(open("train.csv"))))
test_data = np.array(list(csv.reader(open("qaps_2000_cleaned.csv"))))

coarse_train_target = train_data[:,0]
fine_train_target = train_data[:,1]
train_text = train_data[:,2]
coarse_test_target = test_data[:,1]
fine_test_target = test_data[:,2]
test_text = test_data[:,0]

train_target = fine_train_target
test_target = fine_test_target


pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer()),
    ('tfidf',              TfidfTransformer()),
    ('classifier',         OneVsRestClassifier(LinearSVC()))
])


lb = LabelBinarizer()
train_z = lb.fit_transform(train_target)

pipeline.fit(train_text, train_z)
predicted = pipeline.predict(test_text)
predictions = lb.inverse_transform(predicted)
print(len(predictions))


accuracy = accuracy_score(test_target, predictions)
print(accuracy)

