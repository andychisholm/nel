#!/usr/bin/env python
import tempfile
import base64
import pycrfsuite

from time import time
from datetime import datetime
from itertools import izip
from .data import ObjectStore

from nel import logging
log = logging.getLogger()

class NamePartCounts(object):
    """ Entity count model """
    def __init__(self, tag):
        self.mid = 'npcounts[{:}]'.format(tag)
        self.store = ObjectStore.Get('models:' + self.mid)

    def get_part_counts(self, terms):
        if isinstance(terms, (str, unicode)):
            terms = [terms]
        return {t:d['counts'] if d else {} for t, d in izip(terms, self.store.fetch_many(terms))}

class SequenceClassifier(object):
    mid = 'models:seqtaggers'
    def __init__(self, name):
        self.name = name
        dm = ObjectStore.Get(self.mid).fetch(name)
        if dm:
            self.model = self.get_tagger(base64.b64decode(dm['data']))
            from ..features.recognition import SequenceFeatureExtractor
            self.mapper = SequenceFeatureExtractor(**dm.get('params', {}))
        else:
            raise Exception('No sequence classifier found for name (%s) in store' % self.name)

    def tag(self, doc, sequence, state):
        return self.model.tag(self.mapper.sequence_to_instance(doc, sequence, state))

    @staticmethod
    def get_tagger(data):
        with tempfile.NamedTemporaryFile(suffix='ner.model') as f:
            f.write(data)
            f.flush()
            tagger = pycrfsuite.Tagger()
            tagger.open(f.name)
            return tagger

    @classmethod
    def create(cls, name, data, params = {}, metadata = {}):
        timestamp = datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
        log.info('Storing classifier model (%s)...', name)
        ObjectStore.Get(cls.mid).save({
            '_id': name,
            'created_at': timestamp,
            'data': base64.b64encode(data),
            'params': params,
            'metadata': metadata
        })
