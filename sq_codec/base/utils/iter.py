import collections
from itertools import *
from typing import Iterable
import more_itertools


class Rearranger:
    def __init__(self, iterable: Iterable, num: int, re_iter_every_time=True):
        self.iterable = iterable
        self.num = num
        self._iterator = None
        self.re_iter_every_time = re_iter_every_time

    @property
    def iterator(self):
        if self.re_iter_every_time:
            return chain.from_iterable(repeat(self.iterable))
        else:
            if self._iterator is None:
                self._iterator = chain.from_iterable(repeat(self.iterable))
            return self._iterator

    def __iter__(self):
        return islice(self.iterator, self.num)

    def __len__(self):
        return self.num


# supported in python 3.12 [itertools.batched]
def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


consume = more_itertools.consume


class RoundRobin:
    def __init__(self, *iterables):
        self.iterables = iterables

    def __iter__(self):
        return more_itertools.roundrobin(*self.iterables)

    def __len__(self):
        return sum(len(it) for it in self.iterables)
