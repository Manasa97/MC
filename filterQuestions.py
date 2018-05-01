#python app.py -f sample.txt

import re
import os
import sys
import json
import math
import string
import operator
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from collections import defaultdict
from collections import OrderedDict
from collections import Counter

import nltk
#nltk.download("punkt")
#nltk.download("stopwords")
#nltk.download("averaged_perceptron_tagger")


import argparse
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.externals import joblib

from sent_sim import SentenceSimilarity
import utils.linguistic as ling
from utils.file_reader import File_Reader
from utils.file_writer import File_Writer
from utils.feature_construction2 import FeatureConstruction

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

qa_part1_file = "qa.txt"
output_file = "filtered_qa.txt"

def filter():
    """Pipeline of Automatic Question Generation 
    - Args:
        document(str): path of input document
    - Returns:
        question_answers(pandas.dataframe): Q/A/Prediction
    """
    # init classes

    qa = createDataframe()
    print('......')
    print(qa.iloc[0])
    print('--------')
    print(qa.iloc[0].Question)
    fc = FeatureConstruction()
    questionsWithFeatures = fc.extract_feature(qa)
    filteredQA = classify(questionsWithFeatures)
    print(filteredQA)
    qa_output_file = open(output_file, 'wb')
    pickle.dump(filteredQA, qa_output_file)
    #x = pickle.load(open("qa_pickled_file.txt", 'rb'))
    #print(question_answers)
    #return question_answers
 
    


    	


def classify(df):
    """Classification
    - Args:
        df(pandas.dataframe): candidate qa pairs with extracted features 
    - Returns:
        question_answers(pandas.dataframe): Question, Answer, Prediction (label)
    """
    model_path = os.path.dirname(os.path.abspath(__file__)) + '/models/clf.pkl'
    clf = joblib.load(model_path)
    question_answers = df[['Question', 'Answer']]

    X = df.drop(['Answer', 'Question', 'Sentence'], axis=1).as_matrix()
    y = clf.predict(X)

    question_answers['Prediction'] = y
    filtered_qa = question_answers[question_answers['Prediction'] != 0]
    return filtered_qa


def createDataframe():
    qa = pd.DataFrame()
    qa_part1 = open(qa_part1_file,'r').read()
    qa_list_separator = 'Question:'
    qa_separator = 'Answer:'
    qa_list = qa_part1.split(qa_list_separator)
    qa_list.pop(0)
    for i in range(len(qa_list)):
        q, rest = qa_list[i].split(qa_separator)
        a, s = rest.split('Sentence:')
        qa.at[i,'Question'] = q.replace(' .','')
        qa.at[i,'Answer'] = a.replace(' \n','')
        qa.at[i,'Sentence'] = s.replace('.','')
        qa.at[i,'Sentence'] = s.replace('\n','')
    print(qa)
    return qa
	

if __name__ == '__main__':
    filter()

