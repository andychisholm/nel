#!/usr/bin/env python
from ..process.process import Process

FEATURE_SET = set()

class Feature(object):
    @staticmethod
    def Extractable(c):
        FEATURE_SET.add(c)
        return c

    @property
    def id(self):
        # todo: should be abstract to avoid collisions for duplicate feature types instanced by separate sources
        # raise NotImplementedError 
        return self.__class__.__name__

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
