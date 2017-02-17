#!/usr/bin/env python
# load a model and run test sentences against it
from sklearn import linear_model, datasets
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets.base import Bunch
from my_feature_vect import StanceVectorizer
from my_feature_vect import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import pickle
from operator import itemgetter
#import itertools
import csv
import numpy as np
data = Bunch(id=[],truth=[],headline=[],body=[],
categories = ['agree', 'disagree', 'discuss', 'unrelated']
)

with open('count_clf.pkl', 'rb') as f:
    count_clf = pickle.load(f)
""" 
load test data, in the format
body_id, headline, stance, body

"""
predicted_file = open('dev.predicted.csv', 'w', newline='', encoding='utf-8')
predicted_writer = csv.writer(predicted_file, delimiter=',', dialect='unix', quotechar='"',)
predicted_writer.writerow(['Headline', 'Body ID', 'Stance'])
truth_file = open('dev.truth.csv', 'w', newline='', encoding='utf-8')
truth_writer = csv.writer(truth_file, delimiter=',', dialect='unix', quotechar='"',)
truth_writer.writerow(['Headline', 'Body ID', 'Stance'])
categories = ['agree', 'disagree', 'discuss', 'unrelated']
with open('dev.csv', newline='',encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        #prediction = categories[count_clf.predict(tuple([row[1],row[3]]))[0]]
        prediction = categories[count_clf.predict(tuple([row[1]+row[3]]))[0]]
        predicted_writer.writerow([row[1], row[0], prediction])
        truth_writer.writerow([row[1], row[0], row[2]])
