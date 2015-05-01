import numpy
import random

from .train import TrainMentionClassifier

import logging
log = logging.getLogger()

class TrainLinearResolver(TrainMentionClassifier):
    """ Trains a linear nil resolver over a corpus of documents. """
    NIL_CLS = '0'
    NON_NIL_CLS = '1'
    def __init__(self, **kwargs):
        self.ranking_feature = kwargs.pop('ranker')
        super(TrainLinearResolver, self).__init__(**kwargs)
        self.hparams['fit_intercept'] = True

    def iter_instances(self, docs):
        toggle = True

        for doc in docs:
            for chain in doc.chains:
                if not chain.candidates:
                    # skip mentions without candidates
                    continue

                top_candidate = sorted(chain.candidates, key=lambda c: c.features[self.ranking_feature], reverse=True)[0]

                # todo: adjust class weights
                if chain.resolution:
                    # only learn postitive instances where the ranker correctly resolves the mention
                    # todo: experiment with training NILs regardless of whether the ranker got it right
                    if top_candidate.id == chain.resolution.id:
                        yield top_candidate.fv, self.NON_NIL_CLS
                else:
                    yield top_candidate.fv, self.NIL_CLS

    @classmethod
    def add_arguments(cls, p):
        super(TrainLinearResolver, cls).add_arguments(p)
        p.add_argument('--ranker', metavar='RANKING_FEATURE_ID')
        return p

