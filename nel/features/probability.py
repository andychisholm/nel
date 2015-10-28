#!/usr/bin/env python
import math
from .feature import Feature
from ..model import disambiguation

import logging
log = logging.getLogger()

class LogFeature(Feature):    
    def compute(self, doc, chain, candidate, state):
        return math.log(self.compute_raw(doc, chain, candidate, state))

    def compute_raw(self, doc, chain, candidate, state):
        raise NotImplementedError

@Feature.Extractable
class EntityProbability(LogFeature):
    """ Entity prior probability. """
    def __init__(self, entity_prior_model_tag):
        self.tag = entity_prior_model_tag
        self.em = disambiguation.EntityCounts(self.tag)

    def compute_raw(self, doc, chain, candidate, state):
        return self.em.count(candidate.id) + 0.1

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('entity_prior_model_tag', metavar='ENTITY_MODEL_TAG')
        p.set_defaults(featurecls=cls)
        return p

@Feature.Extractable
class NameProbability(LogFeature):
    """ Conditional probability of an entity given the mentioned name. """
    def __init__(self, name_model_tag):
        self.tag = name_model_tag
        self.npm = disambiguation.NameProbability(self.tag)

    def compute_doc_state(self, doc):
        sfs = set(m.text.lower() for c in doc.chains for m in c.mentions)
        probs_by_sf = self.npm.get_probs_for_names(sfs)

        probs_by_chain = {}
        for chain in doc.chains:
            names = set(m.text.lower() for m in chain.mentions)
            names = sorted(names, key=len, reverse=True)
            sf = None
            for name in names:
                # todo: refactor - depending on the store, this can be a fairly wasteful lookup
                if probs_by_sf[name]:
                    sf = name
                    break
            # we take the longest non-zero name, or the longest name if all are zeros
            probs_by_chain[chain] = probs_by_sf[sf or names[0]]

        return probs_by_chain

    def compute_raw(self, doc, chain, candidate, state):
        return state[chain].get(candidate.id, 1e-10)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('name_model_tag', metavar='NAME_MODEL_TAG')
        p.set_defaults(featurecls=cls)
        return p
