from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
from random import shuffle, seed
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
import copy
from sklearn.decomposition import LatentDirichletAllocation

ROOT=os.getcwd()
seed(1)
np.random.seed(1)

def calculate(topics=[], lis=[], count1=0):
    count = 0
    for i in topics:
        if i in lis:
            count += 1
    if count >= count1:
        return count
    else:
        return 0


def recursion(topic=[], index=0, count1=0):
    count = 0
    global data
    # print(data)
    # print(topics)
    d = copy.deepcopy(data)
    d.pop(index)
    for l, m in enumerate(d):
        # print(m)
        for x, y in enumerate(m):
            if calculate(topics=topic, lis=y, count1=count1) != 0:
                count += 1
                break
                # data[index+l+1].pop(x)
    return count


data = []

def jaccard(a, score_topics=[], term=0,runs=9):
    global data
    data = score_topics
    j_score = []
    for i, j in enumerate(data):
        for l, m in enumerate(j):
            sum = recursion(topic=m, index=i, count1=term)
            if sum != 0:
                j_score.append(sum / float(runs))
    if len(j_score) == 0:
        j_score = [0]
    Y = sorted(j_score)
    return Y[int(len(Y) / 2)]


def get_top_words(model, feature_names, n_top_words, i=0):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        li = []
        for j in topic.argsort()[:-n_top_words - 1:-1]:
            li.append(feature_names[j].encode('ascii', 'ignore'))
        topics.append(li)
    return topics


def readfile1(filename=''):
    dict = []
    with open(filename, 'r') as f:
        for doc in f.readlines():
            try:
                row = doc.lower().strip()
                dict.append(row)
            except:
                pass
    return dict


def _test_LDA(data_samples=[], term=7, random_state=1,max_iter=100,runs=10, **l):
    topics = []
    for i in range(runs):
        shuffle(data_samples)

        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tf = tf_vectorizer.fit_transform(data_samples)

        lda1 = LatentDirichletAllocation(max_iter=max_iter,learning_method='online',random_state=random_state,**l)

        lda1.fit_transform(tf)
        tf_feature_names = tf_vectorizer.get_feature_names()
        topics.append(get_top_words(lda1, tf_feature_names, term, i=i))
    return topics


def ldavem(*x, **r):

    l = np.asarray(x)
    n_components = l[0]['n_components']
    doc_topic_prior=l[0]['doc_topic_prior']
    topic_word_prior=l[0]['topic_word_prior']
    run=10
    topics = _test_LDA( data_samples=r['data_samples'],term=int(r['term'])
                        ,random_state=r['random_state'],max_iter=r['max_iter'],runs=run, n_components=n_components,
                       doc_topic_prior=doc_topic_prior,topic_word_prior=topic_word_prior)

    a = jaccard(n_components, score_topics=topics, term=int(r['term']), runs=run-1)
    return a
