#!/usr/bin/env python
from sklearn import linear_model, datasets
#from sklearn.feature_extraction.text import CountVectorizer
from my_feature_vect import StanceVectorizer
from my_feature_vect import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
#import itertools
import csv
import numpy as np
data = []
truth = []
categories = ['agree', 'disagree', 'discuss', 'unrelated']

# load training data, in the format
# body_id, headline, stance, body
with open('train.csv', newline='',encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        data.append(row[3])
        truth.append(categories.index(row[2]))
# create ngrams from the data
# the default is just ngrams
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data)
print(count_vect.get_feature_names()[1336])
#print(X_train_counts[0])
#print(X_train_counts.shape)

#normalize counts
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
cx = X_train_tf.tocoo()
#for i,j,v in zip(cx.row,cx.col,cx.data):
#    print(i,j,v)
print(X_train_tf.getrow(0))
print("max")
#print(X_train_counts.getrow(214)[:10])
#print(X_train_tf.shape)
#print(data[0])

# try a naive bayes classifier
clf = MultinomialNB().fit(X_train_tf, truth)

# now try a prediction
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tf_transformer.transform(X_new_counts)
feature_names = count_vect.get_feature_names()
for i,label in enumerate(clf.classes_):
    top10 = np.argsort(clf.coef_[i])[-10:]
    print("%s: %s" % (categories[label],
                  " ".join(feature_names[j] for j in top10)))
exit()
predicted = clf.predict(X_new_tfidf)
print(predicted)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, categories[category]))

# now let's try a support vector machine (SVM)
print("pipeline method")
text_clf = Pipeline([('vect', StanceVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, n_iter=5, random_state=42)),
])
_ = text_clf.fit(data, truth)
#print(text_clf.named_steps['vect'].vocabulary_.get('food'))
predicted = text_clf.predict(docs_new)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, categories[category]))
#predicted = text_clf.predict(docs_test)
#np.mean(predicted == twenty_test.target)

