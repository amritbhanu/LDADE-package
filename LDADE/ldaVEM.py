from __future__ import print_function, division

__author__ = 'amrit'

import os
from random import randint, random, seed, shuffle, sample
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from collections import Counter
import copy
import time
import numpy as np


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
    labels = []  # ,6,7,8,9]
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

    # print(file_data)
    X = range(len(labels))
    Y_median = []
    Y_iqr = []
    for feature in labels:
        Y = file_data['citemap'][feature]
        Y = sorted(Y)
        return Y[int(len(Y) / 2)]


def get_top_words(model, path1, feature_names, n_top_words, i=0, file1=''):
    topics = []
    fo = open(path1, 'a+')
    fo.write("Run: " + str(i) + "\n")
    for topic_idx, topic in enumerate(model.components_):
        str1 = ''
        fo.write("Topic " + str(topic_idx) + ": ")
        for j in topic.argsort()[:-n_top_words - 1:-1]:
            str1 += feature_names[j] + " "
            fo.write(feature_names[j] + " ")
        topics.append(str1)
        fo.write("\n")
    fo.close()
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


def _test_LDA(l, path1, file='', data_samples=[]):
    n_topics = 10
    n_top_words = 10

    fileB = []
    fileB.append(file)

    topics = []
    for j, file1 in enumerate(fileB):
        for i in range(10):
            # shuffling the list
            shuffle(data_samples)

            tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
            tf = tf_vectorizer.fit_transform(data_samples)

            lda = LatentDirichletAllocation(n_components=int(l[0]), doc_topic_prior=l[1],
                                            topic_word_prior=l[2],
                                            learning_method='online',
                                            learning_decay=0.7, learning_offset=50.,
                                            max_iter=10)
            lda.fit_transform(tf)

            # print("done in %0.3fs." % (time() - t0))
            tf_feature_names = tf_vectorizer.get_feature_names()
            topics.extend(get_top_words(lda, path1, tf_feature_names, n_top_words, i=i,
                                        file1=file1))
    return topics


def main(*x, **r):
    # 1st r
    start_time = time.time()
    base = '/share/aagrawa8/Data/results/'
    path = os.path.join(base, 'tuned_VEM', r['file'], str(r['term']))
    if not os.path.exists(path):
        os.makedirs(path)
    l = np.asarray(x)
    # print(l)
    b = int(l[0])
    path1 = path + "/K_" + str(b) + "_a_" + str(l[1]) + "_b_" + str(l[2]) + ".txt"
    with open(path1, "w") as f:
        f.truncate()

    topics = _test_LDA(l, path1, file=r['file'], data_samples=r['data_samples'])
    # 2nd method
    # another_method()
    a = jaccard(int(l[0]), score_topics=topics, term=r['term'])
    fo = open(path1, 'a+')
    fo.write("\nRuntime: --- %s seconds ---\n" % (time.time() - start_time))
    fo.write("\nScore: " + str(a))
    fo.close()
    return a
