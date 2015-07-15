import numpy
import random

from ..model import model
from ..features import mapping
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
        self.mapping ='ZeroMeanUnitVarianceMapper'

        self.hparams['fit_intercept'] = True
        self.hparams['C'] = 0.1
        self.hparams['class_weight'] = 'auto'

    def iter_instances(self, docs):
        toggle = True

        for doc in docs:
            for chain in doc.chains:
                if not chain.candidates:
                    # skip mentions without candidates
                    continue

                for _ in chain.mentions:
                    if chain.resolution:
                        for c in chain.candidates:
                            if c.id == chain.resolution.id:
                                yield c.fv, self.NON_NIL_CLS
                                break
                    else:
                        for c in sorted(chain.candidates, key=lambda c: c.features[self.ranking_feature], reverse=True)[:5]:
                            yield c.fv, self.NIL_CLS
                    break

    @classmethod
    def add_arguments(cls, p):
        super(TrainLinearResolver, cls).add_arguments(p)
        p.add_argument('--ranker', metavar='RANKING_FEATURE_ID')
        return p

