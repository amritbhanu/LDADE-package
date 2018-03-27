from __future__ import print_function, division

__author__ = 'amrit'

import sys
sys.dont_write_bytecode = True
from ldaVEM import *

from DE import DE
from collections import OrderedDict


class BaseConfig:
    """
    This is the basic version of user config class
    Providing all the Basic Util inside this class
    NO EXTRA DATA CONFIG INSIDE
    """
    def __init__(self):
        pass

    def __getitem__(self, x):
        return self.__dict__[x]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


class UserNullConfig(BaseConfig):
    """
    This is the inherited version of basic user config class
    ALL THE DATA INSIDE THIS CLASS IS NULL
    IT PROVIDES ALL THE INTERFACE VARIABLES
    """
    def __init__(self):
        BaseConfig.__init__(self)
        self.__dict__.update(F=None,
                             CR=None,
                             NP=None,
                             GEN=None,
                             Goal=None,
                             data_samples=None,
                             terms=None,
                             fitness=None,
                             max_iter=None,
                             termination=None,
                             random_state=None,
                             learners_para=None,
                             learners_para_bounds=None,
                             learners_para_categories=None)


class UserTestConfig(BaseConfig):
    """
    This is the inherited version of basic user config class
    PRE-WRITTEN FOR THE USE OF TESTING CLASS
    CONTAINING ALL THE NECESSARY CONFIG INSIDE
    """
    def __init__(self):
        BaseConfig.__init__(self)
        self.__dict__.update(F=0.3,
                             CR=0.7,
                             NP=10,
                             GEN=2,
                             Goal="Max",
                             data_samples=None,
                             term=7,
                             fitness=ldavem,
                             max_iter=10,
                             termination="Early",
                             random_state=1,
                             learners_para=OrderedDict([("n_components", 10),
                                                        ("doc_topic_prior", 0.1),
                                                        ("topic_word_prior", 0.01)]),
                             learners_para_bounds=[(10, 100),
                                                   (0.1, 1),
                                                   (0.01, 1)],
                             learners_para_categories=["integer",
                                                       "continuous",
                                                       "continuous"])


def LDADE(config):
    seed(config["random_state"])
    np.random.seed(config["random_state"])

    de = DE(F=config["F"],
            CR=config["CR"],
            GEN=config["GEN"],
            Goal=config["Goal"],
            termination=config["termination"],
            random_state=config["random_state"])

    v, _ = de.solve(config["fitness"],
                    config["learners_para"],
                    config["learners_para_bounds"],
                    config["learners_para_categories"],
                    term=config["term"],
                    data_samples=config["data_samples"],
                    random_state=config["random_state"],
                    max_iter=config["max_iter"])

    return v.ind, v.fit
