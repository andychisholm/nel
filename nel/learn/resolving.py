import numpy
import random
from scipy.optimize import minimize_scalar
from sklearn.svm import SVC

from ..model.resolution import Classifier
from ..features import mapping
from .train import TrainMentionClassifier

from nel import logging
log = logging.getLogger()

class FitNilThreshold(object):
    """ Fits a threshold that optimises nil accuracy """
    def __init__(self, classifier_id, corpus, feature):
        self.classifier_id = classifier_id
        self.corpus = corpus
        self.feature = feature

    @staticmethod
    def get_objective(pairs, tp, fp):
        def f(x):
            r = (sum(1.0 for s,n in pairs if s-x<0 and n)+tp) / (sum(1.0 for _,n in pairs if n)+tp)
            p = (sum(1.0 for s,n in pairs if s-x<0 and n)+tp) / (sum(1.0 for s,_ in pairs if s-x<0)+tp+fp)
            return -(p*r/(p+r))
        return f

    def __call__(self):
        from pymongo import MongoClient
        from nel.doc import Doc

        docs = [Doc.obj(d) for d in MongoClient().docs[self.corpus].find()]

        log.info('Computing feature statistics over %i documents...', len(docs))
        mapper_params = TrainMentionClassifier.get_mapper_params([self.feature], docs)
        mapper = mapping.FEATURE_MAPPERS['ZeroMeanUnitVarianceMapper'](**mapper_params)
        docs = [mapper(d) for d in docs]

        score_class_pairs = [
            (sorted(c.candidates, key=lambda c: c.fv[0], reverse=True)[0].fv[0], c.resolution == None)
            for d in docs for c in d.chains for m in c.mentions if c.candidates
        ]

        fns = sum(1.0 for d in docs for c in d.chains for m in c.mentions if not c.candidates and c.resolution != None)
        tps = sum(1.0 for d in docs for c in d.chains for m in c.mentions if not c.candidates and c.resolution == None)

        bounds = min(s for s,_ in score_class_pairs), max(s for s,_ in score_class_pairs)
        result = minimize_scalar(self.get_objective(score_class_pairs, tps, fns), method='Bounded', bounds=bounds)

        log.debug('Threshold @ %.2f yields NIL fscore: %.3f', result.x, -result.fun*2)

        log.info('Saving classifier %s...', self.classifier_id)
        Classifier.create(self.classifier_id, {
            'weights': list([1.]),
            'intercept': -result.x,
            'mapping': {
                'name': mapper.__class__.__name__,
                'params': mapper_params
            },
            'corpus': self.corpus,
            'tag': 'dev'
        })

        log.info('Done.')

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('classifier_id', metavar='CLASSIFIER_ID')
        p.add_argument('--corpus', metavar='CORPUS_ID')
        p.add_argument('--feature', metavar='RANKING_FEATURE_ID')
        p.set_defaults(cls=cls)
        return p

class TrainLinearResolver(TrainMentionClassifier):
    """ Trains a linear nil resolver over a corpus of documents. """
    NIL_CLS = '0'
    NON_NIL_CLS = '1'
    def __init__(self, **kwargs):
        self.ranking_feature = kwargs.pop('ranker')
        kwargs['mapping'] = 'ZeroMeanUnitVarianceMapper'
        super(TrainLinearResolver, self).__init__(**kwargs)

    def init_model(self):
        hparams = {
            'kernel': 'rbf',
            'C': 1000.,
            'probability': True
        }

        return SVC(**hparams)

    def iter_instances(self, docs):
        toggle = True

        for doc in docs:
            for chain in doc.chains:
                if not chain.candidates:
                    # skip mentions without candidates
                    continue

                for mention in chain.mentions:
                    if mention.resolution:
                        for c in chain.candidates:
                            if c.id == mention.resolution.id:
                                yield c.fv, self.NON_NIL_CLS
                                break
                    else:
                        for c in sorted(chain.candidates, key=lambda c: c.features[self.ranking_feature], reverse=True)[:10]:
                            yield c.fv, self.NIL_CLS
                    break

    @classmethod
    def add_arguments(cls, p):
        super(TrainLinearResolver, cls).add_arguments(p)
        p.add_argument('--ranker', metavar='RANKING_FEATURE_ID')
        return p

