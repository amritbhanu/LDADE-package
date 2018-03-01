from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True

from LDADE import LDADE
from collections import Counter
import numpy as np
from random import seed

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

seed(1)
np.random.seed(1)

data, _ =readfile1("../data/pitsA.txt")
val=LDADE(term=7, data=data, F=0.3, CR=0.7, NP=10, GEN=2, Goal="Max", termination="Early",random_state=1, max_iter=10)
print(val)
