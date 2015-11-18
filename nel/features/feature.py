#!/usr/bin/env python
from ..process.process import Process

FEATURE_SET = set()

class Feature(object):
    """ Extract features for candidates of chains in a documents. """
    def __init__(self):
        self._tag = None

    @classmethod
    def iter_options(cls):
        return FEATURE_SET

    @staticmethod
    def Extractable(c):
        FEATURE_SET.add(c)
        return c

    @property
    def tag(self):
        return self._tag
    @tag.setter
    def tag(self, value):
        self._tag = value

    @property
    def id(self):
        return self.__class__.__name__ + ('' if not self.tag else '[{}]'.format(self.tag))

    def __call__(self, doc):
        state = self.compute_doc_state(doc)

        for chain in doc.chains:
            for c in chain.candidates:
                c.features[self.id] = self.compute(doc, chain, c, state)

        return doc
    
    def compute_doc_state(self, doc):
        return None

    def compute(self, doc, chain, candidate, state):
        """ Returns value of the feature for the candidate """
        raise NotImplementedError
