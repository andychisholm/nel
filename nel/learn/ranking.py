import numpy
import random
from sklearn.svm import LinearSVC
from .train import TrainMentionClassifier

from nel import logging
log = logging.getLogger()

def sample_by_magnitude(_, negatives, limit):
    # learn from candidates with highest feature vector magnitude
    # given feature vectors are standardised to have 0 mean and unit standard deviation, this  
    # should be like selecting instances with the strongest feature activation
    return sorted(negatives, key=lambda fv: numpy.abs(fv).sum(), reverse=True)[:limit]

def sample_by_mag_difference(positive, negatives, limit):
    return sorted(negatives, key=lambda fv: numpy.abs(positive - fv).sum(), reverse=True)[:limit]

def sample_randomly(_, negatives, limit):
    random.shuffle(negatives)
    return negatives[:limit]

def sample_by_std(_, negatives, limit):
    # diversity in feature activation
    return sorted(negatives, key=numpy.std, reverse=True)[:limit]

class TrainLinearRanker(TrainMentionClassifier):
    """ Trains a linear candidate ranker over a corpus of documents. """
    def __init__(self, name, features):
        super(TrainLinearRanker, self).__init__(name, features, 'PolynomialMapper')

        # todo: parameterise instance selection parameters
        self.sample_instances = sample_by_magnitude
        self.instance_limit = None

    def iter_instances(self, docs):
        return self.iter_pairwise_instances_with_sampling(docs, self.sample_instances, self.instance_limit)

    def init_model(self):
        # todo: parameterise hyperparams
        hparams = {
            'C': 0.01,
            'penalty': 'l2',
            'loss': 'hinge',
            'class_weight': 'balanced',
            'fit_intercept': True
        }

        return LinearSVC(**hparams)

    @staticmethod
    def iter_pairwise_instances_with_sampling(docs, sampler, limit):
        toggle = True

        for doc in docs:
            for chain in doc.chains:
                resolution = chain.mentions[0].resolution
                resolution = resolution.id if resolution else None
                for m in chain.mentions:
                    if (m.resolution.id if m.resolution else None) != resolution:
                        raise Exception('Invalid chain assignment for training set mention...')

                if resolution == None:
                    # ranking model can't learn from NILs
                    continue

                positive = None
                negatives = []
                for c in chain.candidates:
                    if c.id == resolution:
                        positive = c.fv
                    else:
                        negatives.append(c.fv)

                # if gold reolution isn't in the candidate set we can't learn from pairwise differences
                if type(positive) == type(None):
                    # weird if statement here just avoids numpy None comparison madness
                    continue

                for sample in sampler(positive, negatives, limit):
                    a = positive if toggle else sample
                    b = sample if toggle else positive

                    # return X, y
                    yield a - b, 1. if toggle else -1.

                    # toggle class assignment to balance training set
                    toggle = not toggle

    @staticmethod
    def iter_instance_pairs(docs):
        for doc in docs:
            for mention in doc.chains:
                if mention.resolution == None:
                    # ranking model can't learn from NILs
                    continue

                positive = None
                negatives = []
                for c in mention.candidates:
                    if c.id == mention.resolution.id:
                        positive = c.fv
                    else:
                        negatives.append(c.fv)

                if type(positive) == type(None):
                    # weird if statement here just avoids numpy None comparison madness
                    continue

                if negatives:
                    yield positive, negatives
