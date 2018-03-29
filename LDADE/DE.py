from __future__ import print_function, division

__author__ = 'amrit'

import sys
sys.dont_write_bytecode = True

from collections import OrderedDict, namedtuple
from random import random, randint, uniform, seed, choice, sample
import numpy as np

__all__ = ['DE']
Individual = namedtuple('Individual', 'ind fit')


class DE(object):
    def __init__(self,
                 F=0.3,
                 CR=0.7,
                 NP=10,
                 GEN=2,
                 Goal="Max",
                 termination="Early",
                 random_state=1):
        self.F = F
        self.CR = CR
        self.NP = NP
        self.GEN = GEN
        self.GOAL = Goal
        self.termination = termination
        seed(random_state)
        np.random.seed(random_state)

    def initial_pop(self):
        _l = []
        for _ in range(self.NP):
            dic = OrderedDict()
            for i in range(self.para_len):
                dic[self.para_dic.keys()[i]] = self.calls[i](self.bounds[i])
            _l.append(dic)
        return _l
        # return [{self.para_dic.keys()[i]:self.calls[i](self.bounds[i])
        #          for i in range(self.para_len)}
        #          for _ in range(self.NP)]

    # Need a tuple for integer and continuous variable
    #  but need the whole list for category
    def randomisation_functions(self):
        _l = []
        for i in self.para_category:
            if i == 'integer':
                _l.append(self._randint)
            elif i == 'continuous':
                _l.append(self._randuniform)
            elif i == 'categorical':
                _l.append(self._randchoice)
        self.calls = _l

    # Paras will be keyword with default values
    # and bounds would be list of tuples
    def solve(self, fitness, paras=OrderedDict(), bounds=[], category=[], **r):
        self.para_len = len(paras.keys())
        self.para_dic = paras
        self.para_category = category
        self.bounds = bounds
        self.randomisation_functions()
        initial_population = self.initial_pop()

        self.cur_gen = []
        for ind in initial_population:
            self.cur_gen.append(
                Individual(OrderedDict(ind), fitness(ind, **r)))

        if self.termination == 'Early':
            return self.early_termination(fitness, **r)

        else:
            return self.late_termination(fitness, **r)

    def early_termination(self, fitness, **r):
        if self.GEN > 1:
            for x in range(self.GEN):
                trial_generation = []
                for ind in self.cur_gen:
                    v = self._extrapolate(ind)
                    trial_generation.append(
                        Individual(OrderedDict(v), fitness(v, **r)))

                current_generation = self._selection(trial_generation)
                self.cur_gen = current_generation
            best_index = self._get_best_index()
            return self.cur_gen[best_index], self.cur_gen
        else:
            best_index = self._get_best_index()
            return self.cur_gen[best_index], self.cur_gen

    def late_termination(self, fitness, **r):
        lives = 1
        while lives != 0:
            trial_generation = []
            for ind in self.cur_gen:
                v = self._extrapolate(ind)
                trial_generation.append(
                    Individual(OrderedDict(v), fitness(v, **r)))
            current_generation = self._selection(trial_generation)
            if sorted(self.cur_gen) == sorted(current_generation):
                lives = lives - 1
            else:
                self.cur_gen = current_generation

        best_index = self._get_best_index()
        return self.cur_gen[best_index], self.cur_gen

    def _extrapolate(self, ind):
        if random() < self.CR:
            _l = self._select3others()
            mutated = []
            for x, i in enumerate(self.para_category):
                if i == 'continuous':
                    mutated.append(_l[0][_l[0].keys()[x]]
                                   + self.F
                                   * (_l[2][_l[2].keys()[x]]
                                      - _l[2][_l[2].keys()[x]]))
                else:
                    mutated.append(self.calls[x](self.bounds[x]))
            check_mutated = []
            for i in range(self.para_len):
                if self.para_category[i] == 'continuous':
                    check_mutated.append(
                        max(self.bounds[i][0],
                            min(mutated[i], self.bounds[i][1])))
                else:
                    check_mutated.append(mutated[i])
            dic = OrderedDict()
            for i in range(self.para_len):
                dic[self.para_dic.keys()[i]] = check_mutated[i]
            return dic
        else:
            dic = OrderedDict()
            for i in range(self.para_len):
                key = self.para_dic.keys()[i]
                dic[self.para_dic.keys()[i]] = ind.ind[key]
            return dic

    def _select3others(self):
        _l = []
        val = sample(self.cur_gen, 3)
        for a in val:
            _l.append(a.ind)
        return _l

    def _selection(self, trial_generation):
        generation = []

        for a, b in zip(self.cur_gen, trial_generation):
            if self.GOAL == 'Max':
                if a.fit >= b.fit:
                    generation.append(a)
                else:
                    generation.append(b)
            else:
                if a.fit <= b.fit:
                    generation.append(a)
                else:
                    generation.append(b)

        return generation

    def _get_best_index(self):
        if self.GOAL == 'Max':
            best = 0
            max_fitness = -float("inf")
            for i, x in enumerate(self.cur_gen):
                if x.fit >= max_fitness:
                    best = i
                    max_fitness = x.fit
            return best
        else:
            best = 0
            max_fitness = float("inf")
            for i, x in enumerate(self.cur_gen):
                if x.fit <= max_fitness:
                    best = i
                    max_fitness = x.fit
            return best

    def _randint(self, a):
        return randint(*a)

    def _randchoice(self, a):
        return choice(a)

    def _randuniform(self, a):
        return uniform(*a)
