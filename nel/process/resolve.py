#!/usr/bin/env python
from .process import Process
from ..util import spanset_insert

from nel import logging
log = logging.getLogger()

class Resolver(Process):
    """ Assigns resolutions to chains in a document """
    @classmethod
    def iter_options(cls):
        yield FeatureRankResolver
        yield GreedyOverlapResolver

class FeatureRankResolver(Resolver):
    """ Ranks candidates and resolves nils via previously computed feature values """
    def __init__(self, ranking_feature, resolving_feature = None, resolving_threshold = 0.5):
        self.ranking_feature = ranking_feature
        self.resolving_feature = resolving_feature
        self.resolving_threshold = resolving_threshold

    def __call__(self, doc):
        for m in doc.chains:
            m.resolution = None
            if m.candidates:
                top_candidate = sorted(m.candidates, key=lambda c: c.features[self.ranking_feature], reverse=True)[0]
                if not self.resolving_feature or top_candidate.features[self.resolving_feature] > self.resolving_threshold:
                    m.resolution = top_candidate
        return doc

class GreedyOverlapResolver(Resolver):
    def __init__(self, feature):
        self.feature = feature

    def __call__(self, doc):
        """ Resolve overlapping mentions by taking the highest scoring mention span """
        # tracks set of disjoint mention spans in the document
        span_indicies = []

        non_nils = (m for m in doc.chains if m.resolution)
        nils = (m for m in doc.chains if not m.resolution)

        for chain in sorted(non_nils, key=lambda ch: ch.resolution.features[self.feature], reverse=True):
            mentions = []
            for m in sorted(chain.mentions,key=lambda m:len(m.text),reverse=True):
                # only resolve this link if its mention span doesn't overlap with a previous insert
                if spanset_insert(span_indicies, m.begin, m.end - 1):
                    mentions.append(m)
            chain.mentions = mentions

        for chain in nils:
            mentions = []
            for m in sorted(chain.mentions, key=lambda m: len(m.text), reverse=True):
                if spanset_insert(span_indicies, m.begin, m.end - 1):
                    mentions.append(m)
            chain.mentions = mentions

        return doc
