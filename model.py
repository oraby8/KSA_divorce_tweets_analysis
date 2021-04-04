import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn 
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import random

def read_tsv(data_file):
    text_data = list()
    labels = list()
    infile = open(data_file, encoding='utf-8')
    for line in infile:
        if not line.strip():
            continue
        label, text = line.split('\t')
        text_data.append(text)
        labels.append(label)
    return text_data, labels

###############################################################

def load(pos_train_file, neg_train_file, pos_test_file, neg_test_file):
    pos_train_data, pos_train_labels = read_tsv(pos_train_file)
    neg_train_data, neg_train_labels = read_tsv(neg_train_file)

    pos_test_data, pos_test_labels = read_tsv(pos_test_file)
    neg_test_data, neg_test_labels = read_tsv(neg_test_file)
    print('------------------------------------')

    sample_size = 5
    print('{} random train tweets (positive) .... '.format(sample_size))
    print(np.array(random.sample(pos_train_data, sample_size)))
    print('------------------------------------')
    print('{} random train tweets (negative) .... '.format(sample_size))
    print(np.array(random.sample(neg_train_data, sample_size)))
    print('------------------------------------')

    x_train = pos_train_data + neg_train_data
    y_train = pos_train_labels + neg_train_labels

    x_test = pos_test_data + neg_test_data
    y_test = pos_test_labels + neg_test_labels

    print('train data size:{}\ttest data size:{}'.format(len(y_train), len(y_test)))
    print('train data: # of pos:{}\t# of neg:{}\t'.format(y_train.count('pos'), y_train.count('neg')))
    print('test data: # of pos:{}\t# of neg:{}\t'.format(y_test.count('pos'), y_test.count('neg')))
    print('------------------------------------')
    return x_train, y_train, x_test, y_test

###############################################################

def do_sa(n, my_classifier, name, my_data):
    x_train, y_train, x_test, y_test = my_data
    print('parameters')
    print('n grams:', n)
    print('classifier:', my_classifier.__class__.__name__)
    print('------------------------------------')

    pipeline = Pipeline([
        ('vect', TfidfVectorizer(min_df=0.0001, max_df=0.95,
                                 analyzer='word', lowercase=False,
                                 ngram_range=(1, n))),
        ('clf', my_classifier),
    ])

    pipeline.fit(x_train, y_train)
    feature_names = pipeline.named_steps['vect'].get_feature_names()

    y_predicted = pipeline.predict(x_test)

    # Print the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)
    print('# of features:', len(feature_names))
    print('sample of features:', random.sample(feature_names, 40))
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall =  recall_score(y_test, y_predicted, average='weighted')
    return pipeline

ngrams = (1, 2, 3)
results = []
pos_training = 'train_Arabic_tweets_positive_20190413.tsv'
neg_training = 'train_Arabic_tweets_negative_20190413.tsv'

pos_testing = 'test_Arabic_tweets_positive_20190413.tsv'
neg_testing = 'test_Arabic_tweets_negative_20190413.tsv'

classifiers = [LinearSVC(), SVC(), MultinomialNB(),
               BernoulliNB(), SGDClassifier(), DecisionTreeClassifier(max_depth=5),
               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               KNeighborsClassifier(3)
               ]
t_x,t_y,s_x,s_y=dataset
r = do_sa(1, LinearSVC(), 'x', dataset)
import joblib
joblib.dump(r, 'twitter_sentimntal.sav')
