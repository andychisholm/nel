#!/usr/bin/env python
import math
import msgpack
import base64
from itertools import izip

from .data import ObjectStore

from nel import logging
log = logging.getLogger()

def msgpack_deserialize(data):
    return msgpack.loads(base64.b64decode(data))

class EntityCounts(object):
    """ Entity count model """
    def __init__(self, tag):
        self.mid = 'ecounts[{:}]'.format(tag)
        self.store = ObjectStore.Get('models:' + self.mid)

    def iter_counts(self, entities):
        for entity, item in izip(entities, self.store.fetch_many(entities)):
            yield entity, item.get('count', 0) if item else 0

    def count(self, entity):
        item = self.store.fetch(entity) or {}
        return item.get('count', 0)

class NameProbability(object):
    def __init__(self, tag):
        self.mid = 'necounts[{:}]'.format(tag)
        self.store = ObjectStore.Get('models:' + self.mid)

    def iter_name_entities(self):
        for ne in self.store.fetch_all():
            name = ne.pop('_id')
            yield name, ne.iterkeys()

    def iter_probs_for_names(self, names):
        for item in self.store.fetch_many(names):
            if item:
                total = float(item['total'])
                yield {e:c/total for e, c in item['counts'].iteritems()}
            else:
                yield {}

    def iter_counts_for_names(self, names):
        for item in self.store.fetch_many(names):
            yield item['counts'] if item else {}

    def get_counts_for_names(self, names):
        return dict(izip(names, self.iter_counts_for_names(names)))

    def get_probs_for_names(self, names):
        return dict(izip(names, self.iter_probs_for_names(names)))

    def probability(self, name, entity, candidates):
        count = self.get_count(name, entity)

        if count != 0:
            return count / self.get_total(name)
        return 1e-10

    def is_zero(self, name):
        return not self.store.exists(name)

class EntityContext(object):
    def __init__(self, tag):
        self.tag = tag
        self.tfidf_model = ObjectStore.Get('models:tfidfs[' + self.tag + ']', deserializer=msgpack_deserialize)
        self.idf_model = ObjectStore.Get('models:idfs[' + self.tag + ']')

    def get_entity_bows(self, entities):
        return dict(izip(entities, (i['counts'] if i else {} for i in self.tfidf_model.fetch_many(entities))))

    def get_document_bow(self, tfs):
        idfs = (m['idf'] if m else 0. for m in self.idf_model.fetch_many(tfs.iterkeys()))
        return {t:math.sqrt(v)*idf for (t, v), idf in izip(tfs.iteritems(), idfs) if idf > 0}

    def get_entity_bow(self, entity):
        return self.tfidf_model.fetch(entity).get('counts', {})

class EntityEmbeddings(object):
    def __init__(self, tag):
        self.tag = tag
        self.store = ObjectStore.Get('models:embeddings['+self.tag+']', deserializer=msgpack_deserialize)

    def get_embeddings(self, entities):
        return dict(izip(entities, (i['embedding'] if i else None for i in self.store.fetch_many(entities))))

    def __contains__(self, entity):
        return self.store.exists(entity)

    def __getitem__(self, entity):
        item = self.store.fetch(entity)
        return item['embedding'] if item else None
