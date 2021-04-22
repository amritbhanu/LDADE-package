__author__ = 'amrit'

import copy
import math
import multiprocessing as mp
from collections import OrderedDict, namedtuple
from random import choice, randint, random, sample, seed, uniform

import numpy as np

__all__ = ["DE"]
Individual = namedtuple("Individual", "ind fit")


class DE(object):
    def __init__(
        self,
        F=0.3,
        CR=0.7,
        NP=10,
        GEN=2,
        Goal="Max",
        termination="Early",
        random_state=1,
        threads=None,
    ):
        """

        :param F: Mutation Rate to mutate a candidate
        :param CR: Crossover Rate to see if the candidate will survive in the next generation
        :param NP: Number of candidates in the population
        :param GEN: Number of generations to run Differential Evolution for.
        :param Goal: Whether DE is suppose to maximize or minimise the fitness function ("Max" or "Min")
        :param termination: 2 termination criteria, "Early" runs upto given GEN, "Late" runs upto the time there is no
                            improvement in the current generation than the past one.
        :param random_state: random seed
        :param threads: No. of threads to do multi-processing, if not parallelization then do -1,
                    select based on memory availability, then "None"
        """
        self.F = F
        self.CR = CR
        self.NP = NP
        self.GEN = GEN
        self.GOAL = Goal
        self.termination = termination
        self.threads = (
            max(2, math.ceil(mp.cpu_count() * 0.1)) if threads is None else threads
        )
        seed(random_state)
        np.random.seed(random_state)

    def initial_pop(self):
        """
        :return: the initial population of candidates for DE to work with.
        """
        current_generation = []
        for _ in range(self.NP):
            dic = OrderedDict()
            for i in range(self.para_len):
                dic[list(self.para_dic.keys())[i]] = self.calls[i](self.bounds[i])
            current_generation.append(dic)
        return current_generation

    # Need a tuple for integer and continuous variable but need the whole list for category
    def randomisation_functions(self):
        """
        Set the class variable "calls" based on the type of parameter you get.
        """
        random_calls = []
        for i in self.para_category:
            if i == "integer":
                random_calls.append(self._randint)
            elif i == "continuous":
                random_calls.append(self._randuniform)
            elif i == "categorical":
                random_calls.append(self._randchoice)
        self.calls = random_calls

    # Paras will be keyword with default values, and bounds would be list of tuples
    def solve(self, fitness, paras=None, bounds=None, category=None, **r):  # noqa: C901
        """
        :param fitness: (func) The models fitness function for which DE to minimize and maximize
        :param paras: (Ordered Dic) Order of the key-value (parameter name - its default settings) pair to tune.
                        Key should be same as the fitness function keyword parameter name
        :param bounds: (List of tuples) Similar Orderering as "paras" :param, with upper and lower bounds selected
        :param category: (List) Similar ordering as "paras" :param, and the values to be chosen from
                        ["integer", "continuous", "categorical"] case sensitive
        :param r: any additional keyword arguments required by the fitness function.
        :return: the best index and its corresponding sub-optimal score, also the last generations all - suboptimal candidates
        """
        self.para_len = len(paras.keys())
        self.para_dic = paras
        self.para_category = category
        self.bounds = bounds

        if not callable(fitness):
            raise ValueError("Check function {} is not callable".format(fitness))

        if sorted(set(self.para_category)) != ["categorical", "continuous", "integer"]:
            raise ValueError(
                "Parameter categories should be 'categorical', 'continuous', 'integer'. Please choose among a valid "
                "parameter type."
            )
        for i, e in enumerate(self.para_category):
            para_dic = list(self.para_dic.keys())
            if (e == "continuous" or e == "integer") and (
                len(self.bounds[i]) != 2 or type(self.bounds[i]) != tuple
            ):
                raise ValueError(
                    "Check Parameter {} for its category which should be among 'categorical', 'continuous', 'integer' "
                    "or check bounds which should be of tuple.".format(para_dic[i])
                )
            elif (e == "categorical") and (
                type(self.bounds[i]) != tuple or len(self.bounds[i]) == 0
            ):
                raise ValueError(
                    "Check Parameter {} for its category which should be among 'categorical', 'continuous', 'integer' "
                    "or check bounds which should be of tuple.".format(para_dic[i])
                )

        self.randomisation_functions()
        initial_population = self.initial_pop()

        if self.threads == -1:
            self.cur_gen = []
            for ind in initial_population:
                self.cur_gen.append(Individual(OrderedDict(ind), fitness(ind, **r)))
        else:
            self.cur_gen = self.parallelize(
                fitness, initial_population, threads=self.threads, **r
            )

        if self.termination == "Early":
            return self.early_termination(fitness, **r)

        else:
            return self.late_termination(fitness, **r)

    def early_termination(self, fitness, **r):
        """
        :param fitness: (func) The models fitness function for which DE to minimize and maximize
        :param r: any additional keyword arguments required by the fitness function.
        :return: the best index and its corresponding sub-optimal score, also the last generations all - suboptimal candidates
        """
        if self.GEN > 1:
            for _ in range(self.GEN - 1):
                population = [
                    self._extrapolate(i, ind) for i, ind in enumerate(self.cur_gen)
                ]
                if self.threads == -1:
                    trial_generation = [
                        Individual(OrderedDict(v), fitness(v, **r)) for v in population
                    ]
                else:
                    trial_generation = self.parallelize(
                        fitness, population, threads=self.threads, **r
                    )
                current_generation = self._selection(trial_generation)
                self.cur_gen = current_generation
            best_index = self._get_best_index()
            return self.cur_gen[best_index], self.cur_gen
        else:
            best_index = self._get_best_index()
            return self.cur_gen[best_index], self.cur_gen

    def late_termination(self, fitness, **r):
        """
        :param fitness: (func) The models fitness function for which DE to minimize and maximize
        :param r: any additional keyword arguments required by the fitness function.
        :return: the best index and its corresponding sub-optimal score, also the last generations all - suboptimal candidates
        """
        lives = 1
        while lives != 0:
            population = [
                self._extrapolate(i, ind) for i, ind in enumerate(self.cur_gen)
            ]
            if self.threads == -1:
                trial_generation = [
                    Individual(OrderedDict(v), fitness(v, **r)) for v in population
                ]
            else:
                trial_generation = self.parallelize(
                    fitness, population, threads=self.threads, **r
                )
            current_generation = self._selection(trial_generation)
            if sorted(self.cur_gen) == sorted(current_generation):
                lives = lives - 1
            else:
                self.cur_gen = current_generation

        best_index = self._get_best_index()
        return self.cur_gen[best_index], self.cur_gen

    def _extrapolate(self, ix, ind):  # noqa: C901
        """
        :param ix: the current index of the candidate for which you are mutating against.
        :param ind: a candidate to mutate via selecting 3 other candidates.
        :return: returns the mutated candidate.
        """
        if random() < self.CR:  # nosec - B311
            a, b, c = self._select3others(ix)
            mutated = []
            for x, i in enumerate(self.para_category):
                if i == "continuous":
                    new_can = a[list(a.keys())[x]] + self.F * (
                        b[list(b.keys())[x]] - c[list(c.keys())[x]]
                    )
                    mutated.append(new_can)
                elif i == "integer":
                    new_can = a[list(a.keys())[x]] + self.F * (
                        b[list(b.keys())[x]] - c[list(c.keys())[x]]
                    )
                    mutated.append(int(new_can))
                else:
                    mutated.append(self.calls[x](self.bounds[x]))
            check_mutated = []
            for i in range(self.para_len):
                if (
                    self.para_category[i] == "continuous"
                    or self.para_category[i] == "integer"
                ):
                    check_mutated.append(
                        max(self.bounds[i][0], min(mutated[i], self.bounds[i][1]))
                    )
                else:
                    check_mutated.append(mutated[i])
            dic = OrderedDict()
            for i in range(self.para_len):
                dic[list(self.para_dic.keys())[i]] = check_mutated[i]
            return dic
        else:
            dic = OrderedDict()
            for i in range(self.para_len):
                key = list(self.para_dic.keys())[i]
                dic[list(self.para_dic.keys())[i]] = ind.ind[key]
            return dic

    def _select3others(self, ix):
        """
        :param ix: the current index of the candidate for which you are mutating against.
        :return: returns the 3 candidate to be mutated against from the current generation
        """
        sample_gen = copy.deepcopy(self.cur_gen)
        sample_gen.pop(ix)
        val = sample(sample_gen, 3)
        return [a.ind for a in val]

    def _selection(self, trial_generation):
        """see if the new candidates in the current generation is better or in the new generation
        :param trial_generation: evaluated candidates equal to NP for the new generation
        :return: the final generation with the candidates that are optimal out of new and previous generation
        """
        generation = []

        for a, b in zip(self.cur_gen, trial_generation):
            if self.GOAL == "Max":
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
        """
        :return: the index of the optimal candidate from the final generation
        """
        if self.GOAL == "Max":
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

    @staticmethod
    def _randint(a):
        """
        :param a: accepts a tuple of (Min, Max) for which a random integer needs to be selected.
        """
        return randint(*a)  # nosec - B311

    @staticmethod
    def _randchoice(a):
        """
        :param a: accepts a tuple of categorical choices for which a random choice needs to be selected.
        """
        return choice(a)  # nosec - B311

    @staticmethod
    def _randuniform(a):
        """
        :param a: accepts a tuple of (Min, Max) for which a random continuous needs to be selected.
        """
        return uniform(*a)  # nosec - B311

    @staticmethod
    def thread_callable(inprequest):
        fitness = inprequest[0]
        ind = inprequest[1]
        r = inprequest[2]
        return Individual(OrderedDict(ind), fitness(ind, **r))

    # Add Multiprocessor support
    def parallelize(self, fitness, params_list, threads=2, **r):
        """
        Initializes the multiprocessor threads to process a respective function
        :param fitness (obj): function call mapped to each
        :param params_list (list): list of candidates from the population needed by func
        :param threads (int): no of cpus/threads to use for parallel processing
        :return: list of namedtuple with candidate and fitness score.
        """

        def inputGenerator():
            for d in params_list:
                yield (fitness, d, r)

        inputrequests = inputGenerator()
        pool = mp.Pool(threads)
        data = pool.map(self.thread_callable, inputrequests)
        pool.close()
        pool.join()
        return data
