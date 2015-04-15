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

log = logging.getLogger()

ENC = 'utf8'

def get_timestamp():
    return datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')

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
                
                s.save({
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

class NameProbability(object):
    def __init__(self, tag):
        self.mid = 'necounts[{:}]'.format(tag)
        self.store = Store.Get('models:' + self.mid)

    def probability(self, name, entity, candidates):
        ecs = self.store.fetch(name) or {
            'entities': {},
            'total': .0
        }

        if entity in ecs['entities']:
            return ecs['entities'][entity] / float(ecs['total'])
        
        return 1e-10

    def is_zero(self, name):
        return bool(self.store.fetch(name))

    def create(self, name_entity_counts_iter):
        metadata_store = Store.Get('models:meta')
        
        log.info('Flushing existing name-entity counts...')
        self.store.flush()
        metadata_store.delete(self.mid)

        name_count = 0
        name_entity_count = 0
        with self.store.batched_inserter(250000) as s:
            for name, entity_counts in name_entity_counts_iter:
                name_count += 1
                name_entity_count += len(entity_counts)

                s.save({
                    '_id': name,
                    'entities': entity_counts,
                    'total': sum(entity_counts.itervalues())
                })

                if name_count % 250000 == 0:
                    log.debug('Stored %i name->entity counts...', name_entity_count)

        log.info('Stored %i name-entity counts over %i names...', name_entity_count, name_count)

        metadata_store.save({
            '_id': self.mid,
            'name_count': name_count,
            'name_entity_count': name_entity_count,
            'created_at': get_timestamp()
        })

class EntityDescription(object):
    def __init__(self):
        self.store = Store.Get('models:descriptions')

    def get(self, entity):
        return self.store.fetch(entity)

class Candidates(object):
    def __init__(self):
        self.store = Store.Get('models:aliases')

    def search(self, alias):
        res = self.store.fetch(alias.lower())
        return res['entities'] if res else []

    def set(self, alias, entities):
        self.store.save({
            '_id': alias,
            'entities': entities
        })

    @staticmethod
    def normalise_alias(name):
        return name.lower()

    def create(self, entity_name_iterator):
        entities_by_name = defaultdict(set)
        log.info("Preparing name to entity map.")

        for entity, name in entity_name_iterator:
            if len(name) <= 60:
                entities_by_name[self.normalise_alias(name)].add(entity)

        log.info("Dropping existing candidate set...")
        self.store.flush()

        items_iter = entities_by_name.iteritems()
        total = len(entities_by_name)
        
        # todo: refactor to make use of batched inserter class
        batch_sz = 250000
        for i in xrange(0, total, batch_sz):
            log.info("Inserted %i / %i...", i, total)
            self.store.save_many({
                '_id': alias,
                'entities': list(entities)
            } for alias, entities in itertools.islice(items_iter, batch_sz))

        log.info("Done.")

class Redirects(object):
    def __init__(self):
        from .data import MongoStore
        self.store = MongoStore('models','redirects') #Store.Get('models:redirects')
    
    def map(self, entity):
        mapping = self.store.fetch(entity)
        return mapping['target'] if mapping else entity

    def dict(self):
        return {r['_id']:r['target'] for r in self.store.fetch_all()}

class EntityContext(object):
    def __init__(self, tag):
        self.tf_model = EntityTermFrequency(tag)
        self.df_model = TermDocumentFrequency(tag)

    def get_bow(self, tf_iter):
        return {t:math.sqrt(f) * self.df_model.idf(t) for t, f in tf_iter}

    def get_entity_bow(self, entity):
        return self.get_bow(self.tf_model.get_term_counts(entity).iteritems())

    def similarity(self, a, b):
        a_sq = 1.0 * math.sqrt(sum(val * val for val in a.itervalues()))
        b_sq = 1.0 * math.sqrt(sum(val * val for val in b.itervalues()))

        # iterate over the shorter vector
        if len(b) < len(a):
            a, b = b, a

        cossim = sum(value * b.get(index, 0.0) for index, value in a.iteritems())
        cossim /= a_sq * b_sq

        return cossim

class EntityTermFrequency(object):
    def __init__(self, tag):
        self.mid = 'tfs[{:}]'.format(tag)
        self.store = Store.Get('models:' + self.mid)

    @lru_cache(maxsize=100000)
    def get_term_counts(self, entity):
        return (self.store.fetch(entity) or {}).get('tfs', {})

    def create(self, entity_tfs_iterator):
        log.info("Dropping existing tfs...")
        self.store.flush()

        entity_count = 0
        with self.store.batched_inserter(250000) as s:
            for entity, tfs in entity_tfs_iterator:
                s.save({
                    '_id': entity,
                    'tfs': tfs
                })
                entity_count += 1
                if entity_count % 250000 == 0:
                    log.debug('Stored term counts for %i entities...', entity_count)

        log.info('Stored term counts for %i entities.', entity_count)

        metadata_store = Store.Get('models:meta')
        metadata_store.save({
            '_id': self.mid,
            'entity_count': entity_count,
            'created_at': get_timestamp()
        })

class TermDocumentFrequency(object):
    def __init__(self, tag):
        self.mid = 'df[{:}]'.format(tag)
        self.store = Store.Get('models:' + self.mid)
        
        metadata_store = Store.Get('models:meta')
        metadata = metadata_store.fetch(self.mid) or {}
        self.total_docs = metadata.get('total_docs', 0)

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
                s.save({
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

class EntityCooccurrence(object):
    def __init__(self, cooccurrence_counts = None, occurrence_counts = None):
        self.cooccurrence_counts = cooccurrence_counts or {}
        self.occurrence_counts = occurrence_counts or {}

    def cooccurrence_count(self, a, b):
        if a > b:
            a, b = b, a

        # this crazy stacked dictionary method turns out to be much, much more memory/time efficient
        # than using the frozensets of entity pairs for the key.
        # perhaps because memory allocation for the dict blows up when you have lots of keys
        return self.cooccurrence_counts.get(a, {}).get(b, 0)

    def conditional_probability(self, a, b):
        intersection = self.cooccurrence_count(a, b)
        return 0.0 if intersection == 0 else intersection / self.occurrence_counts[b]

    def write(self, path):
        log.info('Writing cooccurrence model to file: %s' % path)
        with open(path, 'wb') as f:
            marshal.dump(self.cooccurrence_counts, f)
            marshal.dump(self.occurrence_counts, f)

    @staticmethod
    def read(path):
        log.info('Reading entity cooccurrence model from file: %s' % path)

        with open(path, 'rb') as f:
            cooccurrence_counts = marshal.load(f)
            occurrence_counts = marshal.load(f)

        return EntityCooccurrence(cooccurrence_counts, occurrence_counts)

class EntityOccurrence(object):
    "Inlinks to an entity article."
    SIZEW = 4527417 # http://en.wikipedia.org/wiki/Wikipedia:Size_of_Wikipedia
    LOGW = math.log(SIZEW)

    def __init__(self, entity_occurences = None):
        self.d = {} if entity_occurences == None else entity_occurences

    def add(self, entity, url):
        self.d.setdefault(entity, set()).add(url)

    def occurrences(self, entity):
        return self.d.get(entity, set())

    @lru_cache(maxsize=1000000)
    def entity_relatedness(self, a, b):
        """ Milne relatedness of two entities. """
        occ_a = self.occurrences(a)
        occ_b = self.occurrences(b)
        occ_common = occ_a.intersection(occ_b)

        try:
            logmax = max(len(occ_a), len(occ_b))
            logmin = min(len(occ_a), len(occ_b))
            logint = len(occ_common)
            return (logmax - logint) / (self.LOGW - logmin)
        except ValueError:
            return 0.0

    def tagme_vote(self, e1, candidate_set):
        """
        Return score for entity e1 based on another anchor's candidates
        (Ferragina & Scaiella, 2010).
        NOTE: For F&S, candidates should have p(entity|name) scores.
        candidate_set - set of (score, entity) tuples
        """
        try:
            return sum([self.entity_relatedness(e1, e2) * score
                        for score, e2 in candidate_set]) / len(candidate_set)
        except ZeroDivisionError:
            return 0.0

    def tagme_score(self, e1, candidate_sets):
        """
        Return total relevance for entity e1 across other anchors
        (Ferragina & Scaiella, 2010).
        candidate_sets - list of candidate sets, unique by anchor
        """
        return sum([self.tagme_vote(e1, a2) for a2 in candidate_sets], 0.0)

    def write(self, path):
        log.info('Writing occurrence model to file: %s' % path)
        marshal.dump(self.d, open(path, 'wb'))

    @staticmethod
    def read(path):
        log.info('Reading entity occurrence model from file: %s' % path)
        return EntityOccurrence(marshal.load(open(path, 'rb')))

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
