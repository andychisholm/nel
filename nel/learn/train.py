import numpy
import random
import math

from sklearn.svm import LinearSVC
from pymongo import MongoClient

from ..doc import Doc
from ..features import mapping

import logging
log = logging.getLogger()

class Train(object):
    """ Train classifier over a set of documents. """
    def __init__(self, corpus, tag, feature, classifier_id):
        if corpus == None:
            raise NotImplementedError
        
        self.corpus_id = corpus
        self.tag_filter = tag
        self.features = feature
        self.mapping = 'PolynomialMapper'
        self.classifier_id = classifier_id

        self.client = MongoClient()
    
    def compute_mapper_params(self, docs): 
        means, stds = [], []
        for f in self.features:
            raw = [c.features[f] for d in docs for m in d.chains for c in m.candidates]
            means.append(numpy.mean(raw))
            stds.append(numpy.std(raw))

        return {
            'features': self.features,
            'means': means,
            'stds': stds
        }

    def get_training_docs(self):
        store = self.client.docs[self.corpus_id]
        
        flt = {}
        if self.tag_filter != None:
            flt['tag'] = self.tag_filter

        # keeping all docs in memory could be problematic for large datasets
        # but simplifies computation of mapper parameters. todo: offline mapper prep
        return [Doc.obj(json) for json in store.find(flt)]
 
    @staticmethod
    def train(docs, train_instance_limit = None, kernel='poly', C=0.0316228, penalty='l2', loss='l1', instance_selection='mag', instance_limit=10):
        # todo: set based on param
        sample_instances = Train.sample_by_magnitude
        
        X, y = [], []
        instance_count = 0
        toggle = True

        for doc in docs:
            for mention in doc.chains:
                if mention.resolution == None:
                    # disambiguation model can't really learn from NILs
                    continue

                positive = None
                negatives = []
                for c in mention.candidates:
                    if c.id == mention.resolution.id:
                        positive = c.fv
                    else:
                        negatives.append(c.fv)
                
                # if gold entity isn't part of the candidate set
                # we can't compute pairwise differences
                if type(positive) == type(None):
                    # avoid numpy None comparison madness
                    continue
                
                instance_count += 1
                for sample in sample_instances(positive, negatives, instance_limit):
                    a = positive if toggle else sample
                    b = sample if toggle else positive
                    
                    X.append(a - b)
                    y.append(1.0 if toggle else -1.0)

                    # toggle class assignment to balance training set
                    toggle = not toggle
                
                # early stopping
                if train_instance_limit != None and instance_count >= train_instance_limit: break
            if train_instance_limit != None and instance_count >= train_instance_limit: break

        log.info('Fitting model... (%i instances)' % (len(y)))
        dual = penalty == 'l2' and loss == 'l1'
        model = LinearSVC(dual=dual, penalty=penalty, C=C, loss=loss)
        model.fit(X, y)

        return model

    @staticmethod
    def sample_by_mag_difference(positive, negatives, limit):
        return sorted(negatives, key=lambda fv: numpy.abs(positive - fv).sum(), reverse=True)[:limit]
    
    @staticmethod
    def sample_randomly(positive, negatives, limit):
        random.shuffle(negatives)
        return negatives[:limit]
    
    @staticmethod
    def sample_by_std(positive, negatives, limit): 
        # diversity in feature activation
        return sorted(negatives, key=lambda fv: numpy.std(fv), reverse=True)[:limit]

    @staticmethod
    def sample_by_magnitude(positive, negatives, limit):
        # learn from candidates with highest feature vector magnitude
        # given feature vectors are standardised to have 0 mean and unit standard deviation, this  
        # should be like selecting instances with the strongest feature activation
        return sorted(negatives, key=lambda fv: numpy.abs(fv).sum(), reverse=True)[:limit]

    def __call__(self):
        """ Train classifier over documents """

        log.info('Fetching training docs (%s-%s)...', self.corpus_id or 'all', self.tag_filter or 'all') 
        docs = self.get_training_docs() 
        
        log.info('Computing feature statistics over %i documents...', len(docs))
        mapper_params = self.compute_mapper_params(docs)
        mapper = mapping.FEATURE_MAPPERS[self.mapping](**mapper_params)

        log.info('Training classifier...')
        model = self.train(mapper(doc) for doc in docs)

        log.info('Storing classifier model (%s)...', self.classifier_id)
        self.client.models.classifiers.save({
            '_id': self.classifier_id,
            'weights': list(model.coef_[0]),
            'mapping': {
                'name': mapper.__class__.__name__,
                'params': mapper_params
            },
            'corpus': self.corpus_id,
            'tag': self.tag_filter
        })

        log.info('Done.')

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('classifier_id', metavar='CLASSIFIER_ID')
        p.add_argument('--corpus', metavar='CORPUS', default=None, required=False)
        p.add_argument('--tag', metavar='TAG', default=None, required=False)
        p.add_argument('--feature', metavar='FEATURE_MODEL', action='append')
        p.set_defaults(cls=cls)
        return p
