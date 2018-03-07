#!/usr/bin/env python
import cPickle as pickle

from .feature import Feature
from .mapping import FEATURE_MAPPERS
from ..model.resolution import Classifier

from nel import logging
log = logging.getLogger()

class ClassifierFeature(Feature):
    """ Computes a feature score based on the output of a classifier over a set of features. """
    def __init__(self, classifier=None, classifier_model_tag=None):
        if (classifier_model_tag is None) == (classifier is None):
            raise Exception('You must provide a classifier_model_tag or classifier instance, but not both.')
        self.classifier = classifier or Classifier.load(classifier_model_tag) 


    def compute_doc_state(self, doc):
        doc = self.classifier.mapper(doc) 

    def predict(self, fv):
        raise NotImplementedError # returns numerical prediction given a vector of features

    def compute(self, doc, chain, candidate, state):
        return self.predict(candidate.fv)

@Feature.Extractable
class ClassifierScore(ClassifierFeature):
    """ Computes a feature score based on the output of the classifier decision function over a set of features. """
    @property
    def id(self):
        return 'ClassifierScore[%s]' % self.classifier.name

    def predict(self, fv):
        return float(self.classifier.model.decision_function([fv]))

@Feature.Extractable
class ClassifierProbability(ClassifierFeature):
    """ Computes a feature score based on the output probability of a classifier over a set of features. """
    @property
    def id(self):
        return 'ClassifierProbability[%s]' % self.classifier.name

    def predict(self, fv):
        return self.classifier.model.predict_proba(fv)[0][1]
    
