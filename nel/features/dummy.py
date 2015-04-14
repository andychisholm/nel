from time import time
from random import Random
from .feature import Feature

@Feature.Extractable
class RandomNumber(Feature):
    """ Computes a random floating point feature value. """
    def __init__(self, seed = None, mean = 0., std = 1.):
        self.tag = None
        
        seed = seed or time()
        self.mean = mean
        self.std = std
        self.rng = Random(seed)

    def compute(self, doc, chain, candidate, state):
        """ Returns a random feature value """
        return self.rng.gauss(self.mean, self.std)

    @classmethod
    def add_arguments(cls, p):
        p.set_defaults(featurecls=cls)
        return p
