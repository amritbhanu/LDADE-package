from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True

from LDADE import LDADE, UserTestConfig
from collections import Counter
import numpy as np
from random import seed
from os import path


def readfile1(filename=''):
    dict = []
    labels=[]
    with open(filename, 'r') as f:
        for doc in f.readlines():
            try:
                row = doc.lower().split(">>>")
                dict.append(row[0].strip())
                labels.append(row[1].strip())
            except:
                pass
    count=Counter(labels)
    import operator
    key = max(count.iteritems(), key=operator.itemgetter(1))[0]
    labels=map(lambda x: 1 if x == key else 0, labels)
    return np.array(dict), np.array(labels)


def demo():
    seed(1)
    np.random.seed(1)
    this_dir, this_filename = path.split(__file__)
    data_path = path.join(this_dir, "data", "pitsA.txt")
    data, _ = readfile1(data_path)
    what = UserTestConfig()
    what["data_samples"] = data
    val = LDADE(what)
    print(val)
