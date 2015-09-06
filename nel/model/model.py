#!/usr/bin/env python
from __future__ import print_function
from collections import Counter, defaultdict
from functools32 import lru_cache

import itertools
import numpy as np
import cPickle as pickle
import math
import logging
from time import time
from datetime import datetime

from .data import ObjectStore, FieldStore, SetStore
from ..features.mapping import FEATURE_MAPPERS

log = logging.getLogger()

ENC = 'utf8'

def get_timestamp():
    return datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')

class Entities(object):
    def __init__(self, tag):
        self.mid = 'entities[{:}]'.format(tag)
        self.store = ObjectStore.Get('models:' + self.mid)

    def get(self, entity):
        return self.store.fetch(entity)

    def iter_ids(self):
        return self.store.iter_ids()

    def iter_entities(self):
        for entity in self.store.fetch_all():
            yield entity['_id'], entity.get('label', ''), entity.get('description', ''), entity.get('aliases', [])

    def create(self, iter_entities):
        metadata_store = ObjectStore.Get('models:meta')

        log.info('Flushing existing entities...')
        self.store.flush()
        metadata_store.delete(self.mid)

        with self.store.batched_inserter(250000) as s:
            entity_count = 0
            for eid, label, description, aliases, _ in iter_entities:
                data = {
                    '_id': eid,
                    'label': label,
                    'description': description,
                    'aliases': aliases
                }
                s.append(data)

                entity_count += 1
                if entity_count % 250000 == 0:
                    log.debug('Stored %i entities...', entity_count)

        log.info('Stored %i entities', entity_count)

        metadata_store.save({
            '_id': self.mid,
            'entity_count': entity_count,
            'created_at': get_timestamp()
        })

class Candidates(object):
    def __init__(self, tag):
        self.mid = 'aliases[{:}]'.format(tag)
        self.store = ObjectStore.Get('models:' + self.mid)

    def search(self, alias):
        res = self.store.fetch(alias.lower())
        return res['entities'] if res else []

    @staticmethod
    def normalise_alias(name):
        return name.lower()

    def create(self, entity_name_iterator):
        entities_by_name = defaultdict(set)
        log.info("Preparing name to entity map.")

        for entity, name in entity_name_iterator:
            if len(name) <= 60:
                entities_by_name[self.normalise_alias(name)].add(entity)

        log.info("Dropping existing candidate sets...")
        self.store.flush()

        log.info("Storing candidate sets for %i names...", len(entities_by_name))
        count = 0
        with self.store.batched_inserter(250000) as s:
            for alias, entities in entities_by_name.iteritems():
                count += 1
                s.append({
                    '_id': alias,
                    'entities': list(entities)
                })
                if count == 10000 or count % 250000 == 0:
                    log.debug('Stored %i candidate sets...', count)

        log.info('Stored %i candidate sets', count)

        metadata_store = ObjectStore.Get('models:meta')
        metadata_store.save({
            '_id': self.mid,
            'total_aliases': count,
            'created_at': get_timestamp()
        })

class Redirects(object):
    def __init__(self, tag, prefetch = False):
        self.mid = 'redirects[{:}]'.format(tag)
        self.store = ObjectStore.Get('models:' + self.mid)

        self.cache = None
        if prefetch:
            log.info('Prefetching redirect map for %s...', tag)
            self.cache = self.dict()

    def map(self, entity):
        if self.cache:
            mapping = self.cache.get(entity, entity)
        else:
            mapping = self.store.fetch(entity)
            mapping = mapping['target'] if mapping else entity

        return mapping

    def dict(self):
        return {r['_id']:r['target'] for r in self.store.fetch_all()}

    def create(self, source_target_iter):
        log.info("Dropping existing redirect set (%s)...", self.mid)
        self.store.flush()

        log.info("Processing mappings...")
        count = 0
        with self.store.batched_inserter(250000) as s:
            for source, target in source_target_iter:
                count += 1
                s.append({
                    '_id': source,
                    'target': target
                })
                if count == 10000 or count % 250000 == 0:
                    log.debug('Stored %i redirect mappings...', count)

        log.info('Stored %i redirects...', count)

        metadata_store = ObjectStore.Get('models:meta')
        metadata_store.save({
            '_id': self.mid,
            'total_redirects': count,
            'created_at': get_timestamp()
        })

class EntityContext(object):
    def __init__(self, tag):
        self.tf_model = EntityTermFrequency(tag, uri='mongodb://localhost')
        self.df_model = TermDocumentFrequency(tag)

    def get_bow(self, tf_iter):
        return {t:math.sqrt(f) * self.df_model.idf(t) for t, f in tf_iter}

    def get_entity_bow(self, entity):
        return self.get_bow(self.tf_model.get_term_counts(entity).iteritems())

    def similarity(self, a, b):
        if not a or not b:
            return 0.0

        a_sq = 1.0 * math.sqrt(sum(val * val for val in a.itervalues()))
        b_sq = 1.0 * math.sqrt(sum(val * val for val in b.itervalues()))

        # iterate over the shorter vector
        if len(b) < len(a):
            a, b = b, a

        cossim = sum(value * b.get(index, 0.0) for index, value in a.iteritems())
        cossim /= a_sq * b_sq

        return cossim

class EntityPrior(object):
    """ Entity prior. """
    def __init__(self, tag):
        self.mid = 'ecounts[{:}]'.format(tag)
        self.store = ObjectStore.Get('models:' + self.mid)

        metadata = ObjectStore.Get('models:meta').fetch(self.mid) or {}
        self.entity_count = metadata.get('count', 0)
        self.total = metadata.get('total', 0)

    def count(self, entity):
        item = self.store.fetch(entity) or {}
        return item.get('count', 0)

    def prior(self, entity):
        "Return score for entity, default 1e-20."
        count = float(self.count(entity))
        return max(1e-20, count / self.total)

    def create(self, entity_count_iter):
        metadata_store = ObjectStore.Get('models:meta')

        log.info('Flushing existing entity counts...')
        self.store.flush()
        metadata_store.delete(self.mid)

        entity_count = 0
        total_count = 0

        log.info('Storing entity counts...')
        with self.store.batched_inserter(250000) as s:
            for entity, count in entity_count_iter:
                entity_count += 1
                total_count += count

                s.append({
                    '_id': entity,
                    'count': count
                })
                if entity_count % 250000 == 0:
                    log.debug('Stored %i entity counts...', entity_count)

        metadata_store.save({
            '_id': self.mid,
            'count': entity_count,
            'total': total_count,
            'created_at': get_timestamp()
        })

class CountModel(object):
    def __init__(self, mid, tag, uri=None):
        self.mid = '{:}[{:}]'.format(mid, tag)
        self.store = FieldStore.Get('models:' + self.mid, uri=uri)

    def merge(self, oid_field_counts_iter):
        self.store.inc_many(oid_field_counts_iter)

    def get_count(self, oid, field):
        value = self.store.fetch_field(oid, field)
        return int(value) if value != None else 0

    def get_counts(self, oid):
        fvs = self.store.fetch_fields(oid)
        return {} if fvs == None else {f:int(v) for f,v in fvs.iteritems()}

    @lru_cache(maxsize=10000)
    def get_total(self, oid):
        return float(sum(self.get_counts(oid).itervalues()))

class NameProbability(CountModel):
    def __init__(self, tag):
        super(NameProbability, self).__init__('necounts', tag)

    def iter_name_entities(self):
        for ne in self.store.fetch_all():
            name = ne.pop('_id')
            yield name, ne.iterkeys()

    def probability(self, name, entity, candidates):
        count = self.get_count(name, entity)

        if count != 0:
            return count / self.get_total(name)
        return 1e-10

    def is_zero(self, name):
        return not self.store.exists(name)

    def merge(self, name_entity_iter):
        ne_counts = defaultdict(Counter)
        for name, entity in name_entity_iter:
            ne_counts[name][entity] += 1

        log.debug('Accumulating %i name->entity counts...', len(name_entity_iter))
        super(NameProbability, self).merge((name, counts) for name, counts in ne_counts.iteritems())

class EntityTermFrequency(CountModel):
    def __init__(self, tag, uri=None):
        super(EntityTermFrequency, self).__init__('tfs', tag, uri=uri)

    @lru_cache(maxsize=100000)
    def get_term_counts(self, entity):
        return self.get_counts(entity)

    def merge(self, entity_term_counts_iter):
        log.debug('Accumulating term counts over %i documents...', len(entity_term_counts_iter))
        super(EntityTermFrequency, self).merge(entity_term_counts_iter)

class TermDocumentFrequency(object):
    def __init__(self, tag):
        self.mid = 'df[{:}]'.format(tag)
        self.store = ObjectStore.Get('models:' + self.mid)

        metadata_store = ObjectStore.Get('models:meta')
        metadata = metadata_store.fetch(self.mid) or {}
        self.total_docs = metadata.get('total_docs', 0)

    def iter_most_frequent_terms(self, limit):
        df_term_iter = ((obj['df'],obj['_id']) for obj in self.store.fetch_all())
        for _, term in sorted(df_term_iter, reverse=True)[:limit]:
            yield term

    @lru_cache(maxsize=500000)
    def idf(self, term):
        dfo = self.store.fetch(term) or {}
        df = float(dfo.get('df', 0))

        return math.log(self.total_docs / (df+1))

    def create(self, total_docs, term_df_iterator):
        log.info("Dropping existing dfs...")
        self.store.flush()

        count = 0
        with self.store.batched_inserter(250000) as s:
            for term, df in term_df_iterator:
                s.append({
                    '_id': term,
                    'df': df
                })
                count += 1
                if count % 250000 == 0:
                    log.debug('Stored document frequencies for %i terms...', count)

        log.info('Stored %i term dfs.', count)

        metadata_store = ObjectStore.Get('models:meta')
        metadata_store.save({
            '_id': self.mid,
            'total_docs': total_docs,
            'created_at': get_timestamp()
        })

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
        log.info('Storing classifier model (%s)...', name)
        ObjectStore.Get(cls.mid).save({
            '_id': name,
            'created_at': get_timestamp(),
            'mapping': mapping,
            'data': data,
            'metadata': metadata
        })

class EntityCooccurrence(CountModel):
    def __init__(self, tag, uri = None):
        super(EntityCooccurrence, self).__init__('eco', tag, uri=uri)

    def cooccurrence_counts(self, entity):
        return self.get_counts(entity)

    def merge(self, entity_sets_iter):
        cm = defaultdict(Counter)

        for entities in entity_sets_iter:
            for i in xrange(0, len(entities)):
                for j in xrange(i, len(entities)):
                    a, b = entities[i], entities[j]
                    cm[a][b] += 1

                    # note that we DO include the i==j case, we're just not counting it twice
                    # this means that the number of times an entity 'cooccurs' with itself is actually
                    # its occurrence count, which is a useful statistic for both P(e) and P(e|e') models
                    if a != b:
                        cm[b][a] += 1

        log.debug('Merging cooccurrence counts for %i entity sets...', len(entity_sets_iter))
        super(EntityCooccurrence, self).merge(cm.iteritems())

class EntityOccurrence(object):
    def __init__(self, tag, store_uri = None):
        self.mid = 'inlinks[{:}]'.format(tag)
        self.store = SetStore.Get('models:' + self.mid, uri=store_uri)

    def merge(self, entity_occurrence_iter):
        self.store.union_many(entity_occurrence_iter)

class WordVectors(object):
    def __init__(self, vocab, vectors):
        self.vocab = vocab
        self.vectors = vectors

    def vocab_size(self):
        return self.vectors.shape[0]

    def vector_size(self):
        return self.vectors.shape[1]

    def word_to_vec(self, word):
        return self.vectors[self.vocab[word]] if word in self.vocab else None

    def write(self, path):
        "Write model to file under path."
        log.debug('Writing %id word vector model for %s words...' % (self.vector_size(), self.vocab_size()))
        with open(path, 'wb') as f:
            marshal.dump(self.vocab, f)
            np.save(f, self.vectors)

    @staticmethod
    def read(model_path):
        "Read model from file."
        log.debug('Loading word vector model from: %s', model_path)
        with open(model_path, 'rb') as f:
            return WordVectors(marshal.load(f), np.load(f))
