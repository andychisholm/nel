#!/usr/bin/env python
import math
from collections import Counter

from .feature import Feature
from ..model.disambiguation import EntityContext

from nel import logging
log = logging.getLogger()

def sparse_cosine_distance(a, b, norm=True):
    if norm:
        a_sq = 1.0 * math.sqrt(sum(val * val for val in a.itervalues()))
        b_sq = 1.0 * math.sqrt(sum(val * val for val in b.itervalues()))

    if len(b) < len(a):
        a, b = b, a

    cossim = sum(value * b.get(index, 0.0) for index, value in a.iteritems())

    if norm:
        cossim /= a_sq * b_sq

    return 1. - cossim

@Feature.Extractable
class BoWMentionContext(Feature):
    """ Bag of Words similarity """
    def __init__(self, context_model_tag):
        self.tag = context_model_tag
        self.ctx_model = EntityContext(self.tag)

    def distance(self, query, entity):
        if not query or not entity:
            return 1.0
        return sparse_cosine_distance(query, entity, norm=False)

    def compute_doc_state(self, doc):
        candidates = set(c.id for chain in doc.chains for c in chain.candidates)
        candidate_bows = self.ctx_model.get_entity_bows(candidates)
        doc_bow = self.ctx_model.get_document_bow(Counter(doc.text.split()))
        return {c:self.distance(doc_bow, entity_bow) for c,entity_bow in candidate_bows.iteritems()}

    def compute(self, doc, chain, candidate, candidate_sim):
        return candidate_sim[candidate.id]

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('context_model_tag', metavar='CONTEXT_MODEL')
        p.set_defaults(featurecls=cls)
        return p
