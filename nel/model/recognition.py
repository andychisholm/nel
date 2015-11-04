#!/usr/bin/env python
import tempfile
import base64
import pycrfsuite

from time import time
from datetime import datetime
from itertools import izip
from .data import ObjectStore
from ..features.recognition import SequenceFeatureExtractor

import logging
log = logging.getLogger()

class SequenceClassifier(object):
    mid = 'models:seqtaggers'
    def __init__(self, name):
        self.name = name
        dm = ObjectStore.Get(self.mid).fetch(name)
        if dm:
            self.model = self.get_tagger(base64.b64decode(dm['data']))
            self.mapper = SequenceFeatureExtractor(**dm.get('params', {}))
        else:
            raise Exception('No sequence classifier found for name (%s) in store', self.name)

    def tag(self, doc, sequence):
        return self.model.tag(self.mapper.sequence_to_instance(doc, sequence))

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

class Candidates(object):
    def __init__(self, tag, limit = 5):
        self.mid = 'necounts[{:}]'.format(tag)
        self.store = ObjectStore.Get('models:' + self.mid)
        self.limit = limit

        self.tei_store = ObjectStore.Get('models:tei['+tag+']')
        self.count_model = ObjectStore.Get('models:ecounts['+tag+']')

    def search(self, alias):
        alias = self.normalise_alias(alias)
        results = self.store.fetch(alias)

        entities = []
        if results:
            entities = [k for k, v in sorted(results['counts'].iteritems(), key=lambda (k,v): v, reverse=True)][:self.limit]
        ctx_entities = []

        #ts = self.tei_store.fetch(alias)
        ts = None
        if ts:
            ctx_entities = set(entities+[c for c, v in ts['entities'].iteritems() ])
            #ctx_entities = set(entities+ts['entities'].keys())
            counts = (i['count'] if i else 0 for i in self.count_model.fetch_many(ctx_entities))
            entity_counts = dict(izip(ctx_entities, counts))
            ctx_entities = sorted(ctx_entities, key=entity_counts.get, reverse=True)[:self.limit]

        return list(set(entities+ctx_entities))

    @staticmethod
    def normalise_alias(name):
        return name.lower().strip()
