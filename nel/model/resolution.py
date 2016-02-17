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
    def __init__(self, name):
        self.name = name
        dm = ObjectStore.Get(self.mid).fetch(name)
        if dm:
            # undo the data store encoding here - it's really binary data
            self.model = pickle.loads(dm['data'].encode('utf-8'))
            self.mapper = FEATURE_MAPPERS[dm['mapping']['name']](**dm['mapping']['params'])
        else:
            raise Exception('No classifier for name (%s) in store', self.name)

    @classmethod
    def create(cls, name, mapping, data, metadata = {}):
        timestamp = datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
        log.info('Storing classifier model (%s)...', name)
        ObjectStore.Get(cls.mid).save({
            '_id': name,
            'created_at': timestamp,
            'mapping': mapping,
            'data': data,
            'metadata': metadata
        })
