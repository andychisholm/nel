#!/usr/bin/env python
import cPickle as pickle
from time import time
from datetime import datetime

from .data import ObjectStore
from ..features.mapping import FEATURE_MAPPERS

from nel import logging
log = logging.getLogger()

class Classifier(object):
    mid = 'models:classifiers'
    def __init__(self, name, mapper, model):
        self.name = name
        self.mapper = mapper
        self.model = model

    @classmethod
    def load(cls, name):
        dm = ObjectStore.Get(cls.mid).fetch(name)
        if not dm:
            raise Exception('No classifier for name (%s) in store', name)
        return cls(**pickle.loads(dm['data']))

    def save(self):
        timestamp = datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
        log.info('Storing classifier model (%s)...', self.name)
        ObjectStore.Get(self.mid).save({
            '_id':self.name,
            'data': pickle.dumps({
                'name': self.name,
                'model': self.model,
                'mapper': self.mapper
            })
        })
        
