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


def jaccard(a, score_topics=[], term=0):
    labels = []
    labels.append(term)
    global data
    l = []
    data = []
    file_data = {}
    for doc in score_topics:
        l.append(doc.split())
    for i in range(0, len(l), int(a)):
        l1 = []
        for j in range(int(a)):
            l1.append(l[i + j])
        data.append(l1)
    dic = {}
    for x in labels:
        j_score = []
        for i, j in enumerate(data):
            for l, m in enumerate(j):
                sum = recursion(topic=m, index=i, count1=x)
                if sum != 0:
                    j_score.append(sum / float(9))
                '''for m,n in enumerate(l):
                    if n in j[]'''
        dic[x] = j_score
        if len(dic[x]) == 0:
            dic[x] = [0]
    file_data['citemap'] = dic
    for feature in labels:
        Y = file_data['citemap'][feature]
        Y = sorted(Y)
        return Y[int(len(Y) / 2)]


def get_top_words(model, feature_names, n_top_words, i=0, file1=''):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        str1 = ''
        for j in topic.argsort()[:-n_top_words - 1:-1]:
            str1 += feature_names[j] + " "
        str1=str(str1.encode('ascii', 'ignore'))
        topics.append(str1)
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


def _test_LDA( file='', data_samples=[], term=7, random_state=1, **l):
    topics = []
    for i in range(10):
        shuffle(data_samples)

        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tf = tf_vectorizer.fit_transform(data_samples)

        lda1 = LatentDirichletAllocation(max_iter=10,learning_method='batch',random_state=random_state,**l)

        lda1.fit_transform(tf)
        tf_feature_names = tf_vectorizer.get_feature_names()
        topics.extend(get_top_words(lda1, tf_feature_names, term, i=i, file1=file))
    return topics


def ldavem(*x, **r):

    l = np.asarray(x)
    n_components = l[0]['n_components']
    doc_topic_prior=l[0]['doc_topic_prior']
    topic_word_prior=l[0]['topic_word_prior']

    topics = _test_LDA( file=r['file'],data_samples=r['data_samples'],term=int(r['term'])
                        ,random_state=r['random_state'],n_components=n_components,
                       doc_topic_prior=doc_topic_prior,topic_word_prior=topic_word_prior)

    a = jaccard(n_components, score_topics=topics, term=int(r['term']))
    return a