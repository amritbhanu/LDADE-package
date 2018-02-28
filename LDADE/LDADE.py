from __future__ import print_function, division

__author__ = 'amrit'

import sys
sys.dont_write_bytecode = True
from ldaVEM import *

from DE import DE
from collections import OrderedDict

fitness=ldavem
learners_para=OrderedDict([("n_components",10),("doc_topic_prior",0.1), ("topic_word_prior",0.01)])
learners_para_bounds=[(10,100), (0.1,1), (0.01,1)]
learners_para_categories=[ "integer", "continuous", "continuous"]

def LDADE(term=7, data=[], F=0.3, CR=0.7, NP=10, GEN=2, Goal="Max", termination="Early", random_state=1, max_iter=100):

    seed(random_state)
    np.random.seed(random_state)

    de = DE(F=F, CR=CR, NP=NP, GEN=GEN, Goal=Goal, termination=termination, random_state=random_state)
    v, _ = de.solve(fitness, learners_para, learners_para_bounds, learners_para_categories,
                    term=7, data_samples=data,random_state=random_state,max_iter=max_iter)

    ## Optimal configuration and its corresponding optimal value.
    return v.ind, v.fit
