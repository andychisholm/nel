#!/usr/bin/env python
import math
from .feature import Feature
from functools32 import lru_cache
from ..model import model

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
    def __init__(self, entity_prior_model_tag):
        self.tag = entity_prior_model_tag
        self.epm = model.EntityPrior(self.tag)

    def candidate_probability(self, doc, mention, candidate, state):
        return self.epm.prior(candidate.id)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('entity_prior_model_tag', metavar='ENTITY_MODEL_TAG')
        p.set_defaults(featurecls=cls)
        return p

@Feature.Extractable
class NameProbability(LogProbabilityFeature):
    """ Entity given Name probability. """
    def __init__(self, name_model_tag):
        self.tag = name_model_tag
        self.npm = model.NameProbability(self.tag)

    def compute_doc_state(self, doc):
        sfs_by_chain = {}
        
        for chain in doc.chains:
            names = set(m.text.lower() for m in chain.mentions)
            names = sorted(names, key=len, reverse=True)
            sf = None
            for name in names:
                # todo: refactor - depending on the store, this can be a fairly wasteful lookup
                if not self.npm.is_zero(name):
                    sf = name
                    break
            # we take the longest non-zero name, or the longest name if all are zeros
            sfs_by_chain[chain] = sf or names[0]
        
        return sfs_by_chain
    
    def candidate_probability(self, doc, chain, candidate, state):
        return self.npm.probability(state[chain], candidate.id, [c.id for c in chain.candidates])

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('name_model_tag', metavar='NAME_MODEL_TAG')
        p.set_defaults(featurecls=cls)
        return p
