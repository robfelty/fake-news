#!/usr/bin/env python
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

""" 
load training data, in the format
body_id, headline, stance, body
Data will contain tuples of headline,body

"""
with open('train.csv', newline='',encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        data.id.append(row[0])
        data.headline.append(row[1])
        data.truth.append(data.categories.index(row[2]))
        data.body.append(row[3])
# create ngrams from the data
# the default is just ngrams
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data.body)
#print(count_vect.get_feature_names()[1336])
#print(X_train_counts[0])
#print(X_train_counts.shape)

#normalize counts
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
cx = X_train_tf.tocoo()
#for i,j,v in zip(cx.row,cx.col,cx.data):
#    print(i,j,v)
#print(X_train_tf.getrow(0))
#print("max")
#print(X_train_counts.getrow(214)[:10])
#print(X_train_tf.shape)
#print(data[0])

# try a naive bayes classifier
clf = MultinomialNB().fit(X_train_tf, data.truth)

# now try a prediction
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
#feature_names = count_vect.get_feature_names()
#print(feature_names)

#for doc, category in zip(docs_new, predicted):
#    print('%r => %s' % (doc, data.categories[category]))

# now let's try a support vector machine (SVM)
print("pipeline method")
text_clf = Pipeline([('vect', StanceVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, n_iter=5, random_state=42)),
])
_ = text_clf.fit(zip(data.headline,data.body), data.truth)
#print(text_clf.named_steps['vect'].vocabulary_.get('food'))
def print_top_features(pipeline, categories, max_features=10, text=None):
    """ Print out the features which have the highest coefficients"""
    clf = pipeline.named_steps['clf']

    if text:
        tvec = pipeline.transform([text]).toarray()
        print(tvec)
        classes = pipeline.predict([text])
        print('%s classified as %s' % (text ,categories[classes[0]]))
    else:
        tvec = clf.coef_
        classes = clf.classes_
    feature_names =pipeline.named_steps['vect'].get_feature_names()
    if len(feature_names) < max_features:
        max_features = len(feature_names)
    for i,label in enumerate(classes):
        top10 = np.argsort(tvec[i])[::-1][:max_features]
        info = categories[label]
        for j in top10:
            info += "%d, %s (%0.4f)" % (j,feature_names[j], tvec[i,j])
        print(info)


def show_most_informative_features(model, text=None, n=20):
    # Extract the vectorizer and the classifier from the pipeline
    vectorizer = model.named_steps['vect']
    classifier = model.named_steps['clf']

    # Check to make sure that we can perform this computation
    if not hasattr(classifier, 'coef_'):
        raise TypeError(
            "Cannot compute most informative features on {}.".format(
                classifier.__class__.__name__
            )
        )

    if text is not None:
        # Compute the coefficients for the text
        tvec = model.transform([text]).toarray()
    else:
        # Otherwise simply use the coefficients
        tvec = classifier.coef_

    # Zip the feature names with the coefs and sort
    coefs = sorted(
        zip(tvec[0], vectorizer.get_feature_names()),
        key=itemgetter(0), reverse=True
    )

    # Get the top n and bottom n coef, name pairs
    topn  = zip(coefs[:n], coefs[:-(n+1):-1])

    # Create the output string to return
    output = []

    # If text, add the predicted value to the output.
    if text is not None:
        output.append("\"{}\"".format(text))
        output.append(
            "Classified as: {}".format(model.predict([text]))
        )
        output.append("")

    # Create two columns with most negative and most positive features.
    for (cp, fnp), (cn, fnn) in topn:
        output.append(
            "{:0.4f}{: >15}    {:0.4f}{: >15}".format(
                cp, fnp, cn, fnn
            )
        )

    return "\n".join(output)

#print_top_features(text_clf,data.categories)
print(show_most_informative_features(text_clf))
#print(show_most_informative_features(text_clf,text=tuple(['burger in this town', 'this is the body text'])))
#sample = ['burgers are old in this town', 'this is actually a new burger in this town']
#print_top_features(text_clf,data.categories,text=sample)
count_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, n_iter=5, random_state=42)),
])
_ = count_clf.fit(data.body, data.truth)
with open('text_clf.pkl', 'wb') as f:
    pickle.dump(text_clf,f,pickle.HIGHEST_PROTOCOL)
with open('count_clf.pkl', 'wb') as f:
    pickle.dump(count_clf,f,pickle.HIGHEST_PROTOCOL)
#print_top_features(count_clf,data.categories,text='burger in this town')
#print(show_most_informative_features(count_clf,text='burger in this town'))
#print(show_most_informative_features(count_clf,text='the of to 2 1995'))
#print(show_most_informative_features(count_clf,text='robert albert felty'))
exit()
print(count_clf.named_steps['vect'].get_feature_names()[87])

predicted = text_clf.predict(docs_new)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, data.categories[category]))
#predicted = text_clf.predict(docs_test)
#np.mean(predicted == twenty_test.target)

