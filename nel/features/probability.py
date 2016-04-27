#!/usr/bin/env python
import math

from .feature import Feature
from ..model import disambiguation
from ..process.candidates import CandidateGenerator

from nel import logging
log = logging.getLogger()

class LogFeature(Feature):    
    def compute(self, doc, chain, candidate, state):
        return math.log(self.compute_raw(doc, chain, candidate, state))

    def compute_raw(self, doc, chain, candidate, state):
        raise NotImplementedError

@Feature.Extractable
class EntityProbability(LogFeature):
    """ Entity prior probability. """
    def __init__(self, entity_model_tag):
        self.tag = entity_model_tag
        self.em = disambiguation.EntityCounts(self.tag)

    def compute_doc_state(self, doc):
        candidates = set(c.id for chain in doc.chains for c in chain.candidates)
        return dict(self.em.iter_counts(candidates))

    def compute_raw(self, doc, chain, candidate, state):
        return state[candidate.id] + 0.1

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('entity_model_tag', metavar='ENTITY_MODEL_TAG')
        p.set_defaults(featurecls=cls)
        return p

@Feature.Extractable
class NameProbability(LogFeature):
    """ Conditional probability of an entity given the mentioned name. """
    def __init__(self, name_model_tag):
        self.tag = name_model_tag
        self.npm = disambiguation.NameProbability(self.tag)

    def compute_doc_state(self, doc):
        sfs = set()
        for c in doc.chains:
            for m in c.mentions:
                sfs.update(CandidateGenerator.get_normalised_forms(m.text))
        return self.npm.get_probs_for_names(sfs)

    def compute_raw(self, doc, chain, candidate, state):
        p = 1e-10
        for m in chain.mentions:
            for sf in CandidateGenerator.get_normalised_forms(m.text):
                if candidate.id in state[sf]:
                    p = max(state[sf][candidate.id], p)
        return p

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('name_model_tag', metavar='NAME_MODEL_TAG')
        p.set_defaults(featurecls=cls)
        return p
