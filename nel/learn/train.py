import numpy
import cPickle as pickle
from pymongo import MongoClient

from ..doc import Doc
from ..features import mapping
from ..model.resolution import Classifier

from nel import logging
log = logging.getLogger()

class TrainMentionClassifier(object):
    """ Abstract class for training an SVM classifier over mentions in a corpus of documents. """
    def __init__(self, tag, features, mapping):
        self.tag = tag
        self.features = features
        self.mapping = mapping

    def init_model(self):
        raise NotImplementedError

    def __call__(self, docs):
        mapper_params = self.get_mapper_params(self.features, docs)
        mapper = self.get_mapper(self.mapping, mapper_params)

        log.info('Building training set, feature mapping = %s...', self.mapping)
        X, y = [], []
        for x, cls in self.iter_instances(mapper(doc) for doc in docs):
            X.append(x)
            y.append(cls)

        X = numpy.array(X)
        y = numpy.array(y)

        log.info('Fitting model over %i instances...', len(y))
        classifier = self.init_model()
        classifier.fit(X, y)

        correct = sum(1.0 for i, _y in enumerate(classifier.predict(X)) if y[i] == _y)
        log.info('Training set pairwise classification: %.1f%% (%i/%i)', correct*100/len(y), int(correct), len(y))

        log.info('Done.')
        return Classifier(self.tag, mapper, classifier)

    def iter_instances(self, docs):
        raise NotImplementedError

    @classmethod
    def get_docs(cls, corpus, tag):
        log.info('Fetching training docs (%s-%s)...', corpus or 'all', tag or 'all')
        store = MongoClient().docs[corpus]

        flt = {}
        if tag != None:
            flt['tag'] = tag

        # keeping all docs in memory could be problematic for large datasets
        # but simplifies computation of mapper parameters. todo: offline mapper prep
        return [Doc.obj(json) for json in store.find(flt)]

    @classmethod
    def get_mapper_params(cls, features, docs):
        log.info('Computing feature statistics over %i documents...', len(docs))
        means, stds = [], []
        for f in features:
            raw = [c.features[f] for d in docs for m in d.chains for c in m.candidates]
            means.append(numpy.mean(raw))
            stds.append(numpy.std(raw))

        return {
            'features': features,
            'means': means,
            'stds': stds
        }

    @classmethod
    def get_mapper(cls, mapper_name, params):
        return mapping.FEATURE_MAPPERS[mapper_name](**params)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('classifier_id', metavar='CLASSIFIER_ID')
        p.add_argument('--corpus', metavar='CORPUS', default=None, required=False)
        p.add_argument('--tag', metavar='TAG', default=None, required=False)
        p.add_argument('--feature', metavar='FEATURE_MODEL', action='append')
        p.set_defaults(cls=cls)
        return p
