#!/usr/bin/env python
from __future__ import print_function
from collections import Counter, defaultdict
from functools32 import lru_cache
from .data import Store

import itertools
import marshal
import operator
import cPickle as pickle
import numpy as np
import math
import os
import logging
import mmap
from time import time
from datetime import datetime

from ..features.mapping import FEATURE_MAPPERS

log = logging.getLogger()

ENC = 'utf8'

def get_timestamp():
    return datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')

class Entities(object):
    def __init__(self, tag):
        self.mid = 'entities[{:}]'.format(tag)
        self.store = Store.Get('models:' + self.mid)

    def get(self, entity):
        return self.store.fetch(entity)

    def iter_ids(self):
        return self.store.iter_ids()

    def iter_entities(self):
        for entity in self.store.fetch_all():
            yield entity['_id'], entity.get('label', ''), entity.get('description', ''), entity.get('aliases', [])

    def create(self, iter_entities):
        metadata_store = Store.Get('models:meta')

        log.info('Flushing existing entities...')
        self.store.flush()
        metadata_store.delete(self.mid)

        with self.store.batched_inserter(250000) as s:
            entity_count = 0
            for eid, label, description, aliases in iter_entities:
                s.append({
                    '_id': eid,
                    'label': label,
                    'description': description,
                    'aliases': aliases
                })

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
        self.store = Store.Get('models:' + self.mid)

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

        metadata_store = Store.Get('models:meta')
        metadata_store.save({
            '_id': self.mid,
            'total_aliases': count,
            'created_at': get_timestamp()
        })

class Redirects(object):
    def __init__(self, tag, prefetch = False):
        self.mid = 'redirects[{:}]'.format(tag)
        self.store = Store.Get('models:' + self.mid)

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

        metadata_store = Store.Get('models:meta')
        metadata_store.save({
            '_id': self.mid,
            'total_redirects': count,
            'created_at': get_timestamp()
        })

class EntityContext(object):
    def __init__(self, tag):
        self.tf_model = EntityTermFrequency(tag)
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
        self.store = Store.Get('models:' + self.mid)

        metadata = Store.Get('models:meta').fetch(self.mid) or {}
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
        metadata_store = Store.Get('models:meta')

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
        self.store = Store.Get('models:' + self.mid, uri=uri, flat=True)

    def merge(self, oid_field_counts_iter):
        self.store.inc_many(oid_field_counts_iter)

    def get_count(self, oid, field):
        return int(self.store.fetch_field(oid, field))

    def get_counts(self, oid):
        obj = self.store.fetch(oid)
        return {} if obj == None else {f:int(v) for f,v in obj.iteritems() if f != '_id'}

class NameProbability(CountModel):
    def __init__(self, tag):
        super(NameProbability, self).__init__('necounts', tag)

    def iter_name_entities(self):
        for ne in self.store.fetch_all():
            name = ne.pop('_id')
            yield name, ne.iterkeys()

    def probability(self, name, entity, candidates):
        ecs = self.store.get_counts(name) or {}

        if entity in ecs:
            return ecs[entity] / float(sum(ecs.itervalues()))

        return 1e-10

    def is_zero(self, name):
        return bool(self.store.fetch(name))

    def merge(self, name_entity_iter):
        ne_counts = defaultdict(Counter)
        for name, entity in name_entity_iter:
            ne_counts[name][entity] += 1

        log.debug('Accumulating %i name->entity counts...', len(name_entity_iter))
        super(NameProbability, self).merge(ne_counts.iteritems())

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
        self.store = Store.Get('models:' + self.mid)
        
        metadata_store = Store.Get('models:meta')
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

        metadata_store = Store.Get('models:meta')
        metadata_store.save({
            '_id': self.mid,
            'total_docs': total_docs,
            'created_at': get_timestamp()
        })

class LinearClassifier(object):
    mid = 'models:classifiers'
    def __init__(self, name):
        self.name = name
        model = Store.Get(self.mid).fetch(name)
        if model:
            self.weights = np.array(model['weights'])
            self.intercept = model['intercept']
            self.mapper = FEATURE_MAPPERS[model['mapping']['name']](**model['mapping']['params'])
        else:
            raise SystemException('No classifier for name (%s) in store', self.name)

    def score(self, candidate):
        return np.dot(candidate.fv, self.weights) + self.intercept

    @classmethod
    def create(cls, name, model):
        log.info('Storing linear classifier model (%s)...', name)
        model = dict(model)
        model['_id'] = name
        model['created_at'] = get_timestamp()
        Store.Get(cls.mid).save(model)

class EntityCooccurrence(object):
    def __init__(self, tag, store_uri = None):
        self.mid = 'eco[{:}]'.format(tag)
        self.store = Store.Get('models:' + self.mid, flat=True, uri=store_uri)

    def _obj_to_cooccurrence_counts(self, obj):
        return {k:float(v) for k, v in obj if k != '_id'}

    def cooccurrence_counts(self, entity):
        return self._obj_to_cooccurrence_counts(self.store.fetch(entity))

    def iter_cooccurrence_counts(self, entities):
        for obj in self.store.fetch_many(entities):
            yield self._obj_to_cooccurrence_counts(obj)

    def map_entity_to_field(self, entity):
        return entity.replace('.', u'\u2024').replace('$', u'\uff04')
    def map_field_to_entity(self, field):
        return entity.replace(u'\u2024', '.').replace(u'\uff04', '$')

    def merge(self, entity_sets_iter):
        cm = defaultdict(lambda: defaultdict(int))

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
        self.store.inc_many(
            (a, ((self.map_entity_to_field(b), count) for b, count in bs.iteritems()))
            for a, bs in cm.iteritems())
        log.debug('Done...')

class EntityOccurrence(object):
    def __init__(self):
        # todo: needs re-implementation for store api
        raise NotImplementedError

class Links(object):
    "Outlinks from entity article."
    def __init__(self, outlinks = None):
        self.d = {} if outlinks == None else outlinks

    def get(self, source):
        return self.d.get(source, set())

    def update(self, source, target):
        self.d.setdefault(source, set()).add(target)

    def iteritems(self):
        return self.d.iteritems()
    
    def write(self, path):
        log.info('Writing links model to file: %s' % path)
        marshal.dump(self.d, open(path, 'wb'))

    @staticmethod
    def read(path):
        log.info('Reading links model from file: %s' % path)
        return Links(marshal.load(open(path, 'rb')))

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

class EntityTermCounts(object):
    "Model of term counts for a set of entities."
    def __init__(self):
        self.entity_term_counts = defaultdict(dict)

    def update(self, entity, term, count):
        self.entity_term_counts[entity][term] = count

    def get(self, entity, term):
        if entity not in self.entity_term_counts:
            return 0.0
        
        return self.entity_term_counts[entity].get(term, 0.0)

    def get_terms(self, entity):
        return self.entity_term_counts[entity]

    def write(self, path):
        "Write model to file."
        
        log.debug('Writing term frequency model to file: %s ...' % path)

        with open(path, 'wb') as f:
            for entity, counts in self.entity_term_counts.iteritems():
                marshal.dump((entity, counts), f)

    def read(self, path):
        "Read model from file."

        log.debug('Loading raw model from file: %s ...' % path)

        with open(path, 'rb') as fh:
            while True:
                try:
                    entity, counts = marshal.load(fh)
                    self.entity_term_counts[entity] = counts
                except EOFError: break

class mmdict(object):
    def __init__(self, path):
        self.path = path
        self.index = {}
        
        index_path = self.path + '.index'
        log.debug('Loading mmap store: %s ...' % index_path)
        with open(index_path, 'rb') as f:
            while True:
                try:
                    key, offset = self.deserialise(f)
                    self.index[key] = offset
                except EOFError: break

        self.data_file = open(path + '.data', 'rb')
        self.data_mmap = mmap.mmap(self.data_file.fileno(), 0, prot=mmap.PROT_READ)
    
    @staticmethod
    def serialise(obj, f):
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def deserialise(f):
        return pickle.load(f)

    @staticmethod
    def static_itervalues(path):
        with open(path + '.data', 'rb') as f:
            while True:
                try:
                    yield mmdict.deserialise(f)
                except EOFError: break      

    def iteritems(self):
        sorted_idx = sorted(self.index.iteritems(), key=operator.itemgetter(1))
        
        for i, v in enumerate(self.itervalues()):
            yield (sorted_idx[i][0], v)

    def iterkeys(self):
        return self.index.iterkeys()

    def itervalues(self):
        self.data_mmap.seek(0)
        while True:
            try:
                yield self.deserialise(self.data_mmap)
            except EOFError: break

    def __len__(self):
        return len(self.index)

    def __contains__(self, key):
        return key in self.index

    @lru_cache(maxsize=20000)
    def __getitem__(self, key):
        if key not in self:
            return None

        self.data_mmap.seek(self.index[key])
        return self.deserialise(self.data_mmap)

    def __enter__(self):
        return self

    def close(self):
        if hasattr(self, 'data_mmap') and self.data_mmap != None:
            self.data_mmap.close()
        if hasattr(self, 'data_file') and self.data_file != None:
            self.data_file.close()

    def __exit__(self, type, value, traceback):
        self.close()

    def __del__(self):
        self.close()

    @staticmethod
    def write(path, iter_kvs):
        with open(path + '.index','wb') as f_index, open(path + '.data', 'wb') as f_data:
            for key, value in iter_kvs:
                mmdict.serialise((key,f_data.tell()), f_index)
                mmdict.serialise(value, f_data)

class EntityMentionContexts(object):
    "Simple model for entity mentions and context."
    def __init__(self):
        self.entity_mentions = dict()

    def add(self, entity, source, left, sf, right):
        mention = (source, left, sf, right)
        self.entity_mentions.setdefault(entity, []).append(mention)

    def entity_count(self):
        return len(self.entity_mentions)

    def iter_entities(self):
        return self.entity_mentions.iterkeys()

    def mention_count(self):
        return sum(len(ms) for ms in self.entity_mentions.itervalues())

    def mentions(self, entity):
        return self.entity_mentions.get(entity, [])

    def iteritems(self):
        for entity, mentions in self.entity_mentions.iteritems():
            for mention in mentions:
                yield (entity, mention)

    def get_entities_by_surface_form(self):
        entities_by_name = defaultdict(set)

        for entity, mentions in self.entity_mentions.iteritems():
            for _, _, name, _ in mentions:
                entities_by_name[name.lower()].add(entity)

        return entities_by_name

    def write(self, path):
        "Write model to file."

        log.debug(
            'Writing wikilinks context model (%i entities, %i mentions): %s ...' 
            % (self.entity_count(), self.mention_count(), path))
        
        with open(path, 'wb') as f:
            #marshal.dump(self.term_occurences, f)
            for entity, mentions in self.entity_mentions.iteritems():
                marshal.dump((entity, mentions), f)

    @staticmethod
    def iter_entity_mentions_from_path(path):
        with open(path, 'rb') as fh:
            #marshal.load(fh) # todo: remove after model rebuilt
            while True:
                try:
                    yield marshal.load(fh)
                except EOFError: break

    @staticmethod
    def read(path):
        "Read model from file."
        emc = EntityMentionContexts()

        log.debug('Loading mention context model: %s ...' % path)
        with open(path, 'rb') as fh:
            #marshal.load(fh) # todo: remove after model rebuilt
            while True:
                try:
                    entity, mentions = marshal.load(fh)
                    emc.entity_mentions[entity] = mentions
                except EOFError: break

        return emc
