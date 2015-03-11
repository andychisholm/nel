#!/usr/bin/env python
import math
from .feature import Feature
from functools32 import lru_cache
from ..model.model import Name, Entity

import logging
log = logging.getLogger()

class LogProbabilityFeature(Feature):    
    def compute(self, doc, chain, candidate, state):
        return math.log(self.candidate_probability(doc, chain, candidate, state))

    def candidate_probability(self, doc, mention, candidate, state):
        raise NotImplementedError

@Feature.Extractable
class EntityProbability(LogProbabilityFeature):
    """ Entity prior probability. """
    def __init__(self, entity_model_path):
        self.entity_model = Entity()
        self.entity_model.read(entity_model_path)

    def candidate_probability(self, doc, mention, candidate, state):
        return self.entity_model.score(candidate.id)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('entity_model_path', metavar='ENTITY_MODEL')
        p.set_defaults(featurecls=cls)
        return p

@Feature.Extractable
class NameProbability(LogProbabilityFeature):
    """ Entity given Name probability. """
    def __init__(self, name_model_path):
        self.name_model = Name(lower=True)
        self.name_model.read(name_model_path)

    def compute_doc_state(self, doc):
        sfs_by_chain = {}
        
        for chain in doc.chains:
            names = set(m.text.lower() for m in chain.mentions)
            names = sorted(names, key=len, reverse=True)
            sf = None
            for name in names:
                if name in self.name_model.d:
                    sf = name
                    break
            sfs_by_chain[chain] = sf
        
        return sfs_by_chain
    
    def candidate_probability(self, doc, chain, candidate, state):
        return self.name_model.score(state[chain], candidate.id, [c.id for c in chain.candidates])
        
    @classmethod
    def add_arguments(cls, p):
        p.add_argument('name_model_path', metavar='NAME_MODEL')
        p.set_defaults(featurecls=cls)
        return p
