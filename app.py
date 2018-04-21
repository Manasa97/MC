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
from nltk.tag import StanfordNERTagger

import argparse
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.externals import joblib

from sent_sim import SentenceSimilarity
import utils.linguistic as ling
from utils.file_reader import File_Reader
from utils.file_writer import File_Writer
from utils.gap_selection import GapSelection
from utils.sentence_selection import SentenceSelection
from utils.feature_construction import FeatureConstruction

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


def pipeline(document):
    """Pipeline of Automatic Question Generation 
    - Args:
        document(str): path of input document
    - Returns:
        question_answers(pandas.dataframe): Q/A/Prediction
    """
    # init classes

    
    
    ss = SentenceSelection()
    gs = GapSelection()
    fc = FeatureConstruction()
    st = StanfordNERTagger('/Users/Manasa/Desktop/nlpmodel/stanford-models/' + 'english.muc.7class.distsim.crf.ser.gz')
    

    # build candidate questions, extract features
    
    sentences = ss.prepare_sentences(document)
    candidates = gs.get_candidates(sentences)
    candidates_with_features = fc.extract_feature(candidates)
    question_answers = _classify(candidates_with_features)
    question_answers = replaceGaps(question_answers)
    print(question_answers)
    i = 1
    while i == 1:
        query = input('Enter your question: ').strip()
        matched_question, indexMatchedQuestion = computeSimilarity(query, question_answers)
        matched_answer = question_answers.iloc[indexMatchedQuestion]['Answer']
        print(matched_answer)
        i = int(input("Press 1 to continue, 0 to stop"))
    #return matched_answer
    #return question_answers
    


def computeSimilarity(query, question_answers):
	#for now, get the max sim one
    sm = SentenceSimilarity()
    #query = input('Enter your question: ').strip()
    print(query)
    simDict = {}
    query.replace('which','what')
    query.replace('Which','What')
    for i in range(question_answers.shape[0]):
        if question_answers.iloc[i]['Prediction'] != 0:
            q = question_answers.iloc[i]['Question']
            s = sm.similarity(q, query)
            print('Question- ',q, s)
            simDict[i] = s
    maximum = max(simDict.items(), key = operator.itemgetter(1))
    indexMatchedQuestion = maximum[0]
    maxsimilarity = maximum[1]

    matched_question = question_answers.iloc[indexMatchedQuestion]['Question']
    return matched_question, indexMatchedQuestion


    	


def _classify(df):
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

    for i in range(df.shape[0]):
    	q = df.iloc[i]['Question']
    	q = q.replace("_","")
    	q = q.strip()
    	if len(q.split()) < 2:
    		question_answers.at[i,'Prediction'] = 0

    return question_answers

def replaceGaps(question_answers):

    print('Replacing gaps in questions')
    replacements = {'PERSON':'who', 'LOCATION':'where','O':'what','DATE':'when','PERCENT':'how much','TIME':'when','MONEY':'how much'}
    st7 = StanfordNERTagger(os.environ.get(
    'STANFORD_JARS') + 'english.muc.7class.distsim.crf.ser.gz')
    st3 = StanfordNERTagger(os.environ.get(
    'STANFORD_JARS') + 'english.all.3class.distsim.crf.ser.gz')
    for i in range(question_answers.shape[0]):
        a = question_answers.iloc[i]['Answer']
        q = question_answers.iloc[i]['Question']
        ner = st3.tag(a.split())
        ner = [list(elem) for elem in ner]
        

        for j in ner:
            if j[1] == 'O':
                j[1] = st7.tag(j[0].split())[0][1]

        
        #Take majority tag for now excluding 0s
        tags = [x[1] for x in ner]
        t = [a for a in tags if a != 'O']
        if t is not None and len(t) != 0:
            c = Counter(t)
            majority_tag = c.most_common(1)[0][0]
        else:
            majority_tag = 'O'

        q = q.replace('_',replacements[majority_tag],1)
        q = q.replace('_','')
        question_answers.at[i,'Question'] = q
    return question_answers


	

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--input", help="input document")
    args = parser.parse_args()
    pipeline(args.input)
    #print(pipeline(args.input))
    #print("0: bad question; 1: okay question; 2: good question")
