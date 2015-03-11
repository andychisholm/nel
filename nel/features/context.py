#!/usr/bin/env python
import operator
import re
import math
import numpy
import logging
import random

from functools32 import lru_cache
from collections import Counter

from .feature import Feature
from ..model.model import mmdict, WordVectors

log = logging.getLogger()

from scipy.spatial.distance import cosine as dense_cosine_distance

def sparse_cosine_distance(a, b):
    a_sq = 1.0 * math.sqrt(sum(val * val for val in a.itervalues()))
    b_sq = 1.0 * math.sqrt(sum(val * val for val in b.itervalues()))

    # iterate over the shorter vector
    if len(b) < len(a):
        a, b = b, a

    cossim = sum(value * b.get(index, 0.0) for index, value in a.iteritems())
    cossim /= a_sq * b_sq

    return 1. - cossim

@Feature.Extractable
class BoWMentionContext(Feature):
    """ Bag of Words similarity """
    EXCLUDED_LOW_IDF_TERMS = 40
    def __init__(self, query_ctx_window, idf_model_path, entity_ctx_model_path):
        self.query_window = None if query_ctx_window.lower() == 'full' else int(query_ctx_window)
        idfs = mmdict(idf_model_path)
        self.context_model = mmdict(entity_ctx_model_path)

        vocab_exclude_re = re.compile("^([0-9])+")
        idfs = {t:idf for t,idf in idfs.iteritems() if not vocab_exclude_re.match(t)}
        sorted_idfs = sorted((idf, t) for t, idf in idfs.iteritems())[self.EXCLUDED_LOW_IDF_TERMS:]
        self.vocab = {t:(i,idf) for i, (idf, t) in enumerate(sorted_idfs)}
        
        log.debug("Mention context feature (vocab size=%i) (query window=%s)" % (len(sorted_idfs), str(self.query_window)))
        
    def counts_to_bow(self, counts):
        """ Convert term counts to a TF-IDF weighted Bag of Words """
        bow = {}
        for t, count in counts.iteritems():
            if t in self.vocab:
                _, idf = self.vocab[t]
                bow[t] = math.sqrt(count) * idf

        return bow
    
    @staticmethod
    def ngrams(tokens, n, vocab):
        num_tokens = len(tokens)
        for i in xrange(num_tokens):
            for j in xrange(i+1, min(num_tokens, i+n)+1):
                ngram = '_'.join(tokens[i:j])  # mikolov style
                if ngram in vocab:
                    yield ngram
                else:
                    # back off to a normalised version, might also try stemming
                    normalised = ngram.lower()
                    if normalised in vocab:
                        yield normalised

    def tokens_to_bow_vector(self, tokens):
        return self.counts_to_bow(Counter(self.ngrams(tokens, 3, self.vocab)))

    @lru_cache(maxsize=5000)
    def get_entity_context_vec(self, entity):
        return self.counts_to_bow(self.context_model[entity])

    def default_distance(self):
        return 1.0

    def distance(self, query, entity):
        if not query or not entity:
            return self.default_distance()

        return sparse_cosine_distance(query, entity)

    def compute_doc_state(self, doc):
        return self.tokens_to_bow_vector(doc.text.split())
 
    def compute(self, doc, chain, candidate, doc_bow):
        # figure out what set of query doc tokens we need to inspect
        query_context_vec = doc_bow

        if query_context_vec == None:
            log.debug('No query context for doc: %s' % doc.id)

        distance = self.default_distance()

        if query_context_vec != None and candidate.id in self.context_model:
            entity_context_vec = self.get_entity_context_vec(candidate.id)

            if entity_context_vec != None:
                if len(query_context_vec) != 0 and len(entity_context_vec) != 0:
                    distance = self.distance(query_context_vec, entity_context_vec)

        return distance

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('query_ctx_window', metavar='QRY_CTX_WINDOW')
        p.add_argument('idf_model_path', metavar='IDF_MODEL')
        p.add_argument('entity_ctx_model_path', metavar='ENTITY_CONTEXT_MODEL')
        p.set_defaults(featurecls=cls)
        return p

class DBoWMentionContext(BoWMentionContext):
    def __init__(self, **kwargs):
        self.wordvec_model_path = kwargs.pop('wordvec_model_path')
        self.wordvec_model = WordVectors.read(self.wordvec_model_path)
        super(DBoWMentionContext, self).__init__(**kwargs)

    def default_distance(self):
        return 2.0

    def distance(self, query, entity):
        query = self.bow_to_dbow(query)
        entity = self.bow_to_dbow(entity)
        return dense_cosine_distance(query, entity)

    def combine_representations(self, wordreps):
        raise NotImplementedError

    @lru_cache(maxsize=None)
    def term_vector(self, t, weight):
        return self.wordvec_model.word_to_vec(t)*weight

    def iter_word_reps(self, bow):
        for t, weight in bow.iteritems():
            yield self.term_vector(t, weight)

    def bow_to_dbow(self, bow):
        if len(bow) == 0:
            return None
        else:
            return self.combine_representations(self.iter_word_reps(bow))

    @classmethod
    def add_arguments(cls, p):
        super(DBoWMentionContext, cls).add_arguments(p)
        p.add_argument('wordvec_model_path', metavar='WORD_VECTOR_MODEL')
        return p

@Feature.Extractable
class AvgDBoWMentionContext(DBoWMentionContext):
    """ Averaged Distributed Bag of Words similarity. """
    def combine_representations(self, wordreps):
        bow = numpy.zeros(self.wordvec_model.vector_size(), dtype=numpy.float)
        num = 0.0
        for wr in wordreps: # this is actually faster than numpy.sum
            num += 1
            bow += wr
        return bow / num

@Feature.Extractable
class LLMDBoWMentionContext(DBoWMentionContext):
    """ Lexical level matching over similarity of distributed word representations """
    def distance(self, query, entity):
        q_len = len(query)
        e_len = len(entity)

        if q_len == 0 or e_len == 0:
            return 1.0
        else:
            query = dict(sorted(query.iteritems(), key=operator.itemgetter(1), reverse=True)[:100])
            entity = dict(sorted(entity.iteritems(), key=operator.itemgetter(1), reverse=True)[:100])
            
            query_wrs = list(self.iter_word_reps(query))
            entity_wrs = list(self.iter_word_reps(entity))

            a = entity_wrs if e_len >= q_len else query_wrs
            b = entity_wrs if q_len >  e_len else query_wrs

            total = 0.0
            for wb in b:
                max_sim = 0.0
                for wa in a:
                    sim = dense_cosine_distance(query, entity)
                    #log.debug("SIM: %f", sim)
                    if sim > max_sim:
                        max_sim = sim
                total += max_sim

            #llm_sim = sum(max(self.dense_cosine(wa, wb) for wa in a) for wb in b) / float(len(b))
            #log.debug('LLM SIM: %f - %i - %i - %f' % (total, len(b), len(a), total / len(b)))
            return 1.0 - (total / len(b))

@Feature.Extractable
class MaxDBoWMentionContext(DBoWMentionContext):
    """ Max Distributed Bag of Words similarity. """
    def combine_representations(self, wordreps):
        vec_sz = self.wordvec_model.vector_size()

        bow = numpy.zeros(vec_sz, dtype=numpy.float)
        max_bow = numpy.zeros(vec_sz, dtype=numpy.float)
        min_bow = numpy.zeros(vec_sz, dtype=numpy.float)

        for wr in wordreps:
            max_bow = numpy.fmax(wr, max_bow)
            min_bow = numpy.fmin(wr, min_bow)

        max_bow_abs = numpy.abs(max_bow)
        min_bow_abs = numpy.abs(min_bow)
        for i in xrange(self.wordvec_model.vector_size()):
            if max_bow_abs[i] > min_bow_abs[i]:
                bow[i] = max_bow[i]
            else:
                bow[i] = min_bow[i]
        return bow

import marshal
from gensim.models.doc2vec import Doc2Vec
from ..model.prepare.wordvec import LabeledText
from ..model.prepare.wordvec_utils import add_labeled_texts

logging.getLogger("gensim.models.word2vec").setLevel(logging.WARNING)

@Feature.Extractable
class EmbeddingSimilarity(Feature):
    """ Entity embedding similarity """
    def __init__(self, model_path):
        self.model = Doc2Vec.load(model_path, mmap='r')
        log.info('Vocab size: %i, %i entities', len(self.model.vocab), sum(1 for k in self.model.vocab if k[0] == '~'))
        self.redirects = marshal.load(open('/data0/linking/models/wikipedia.redirect.model', 'rb'))

    def compute_doc_state(self, doc):
        label = '#~' + doc.id
        #x = LabeledText([t.text for t in doc.tokens], [label])
        #print x
        #words = add_labeled_texts(self.model, [x])

        #num_iters = 5
        #for i in xrange(0, num_iters):
        #    self.model.alpha = 0.025 * (num_iters - i) / num_iters + 0.0001 * i / num_iters
        #    self.model.min_alpha = self.model.alpha
        #    self.model.train([x],total_words=words)

        return label

    def compute(self, doc, chain, candidate, state):
        sim = 2.0

        candidate = '~' + self.redirects.get(candidate, candidate)
        
        if candidate in self.model.vocab and state in self.model.vocab:
            sim = 1. - self.model.similarity(candidate, state)
        
        #if doc.id == '947testa CRICKET' and mention.text == u'LEICESTERSHIRE':
        #    log.info('(%s, %s) = %.2f', candidate, state, sim)
        
        return sim

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('model_path', metavar='EMBEDDING_MODEL_PATH')
        p.set_defaults(featurecls=cls)
        return p 

@Feature.Extractable
class CRPClusteredDBoWContext(BoWMentionContext):
    """ Similarity over document vectors partitioned via a Chinese Restaurant Process """
    def __init__(self, **kwargs):
        self.wordvec_model_path = kwargs.pop('wordvec_model_path')
        self.wordvec_model = WordVectors.read(self.wordvec_model_path)
        super(CRPClusteredDBoWContext, self).__init__(**kwargs)

    def default_distance(self):
        return -1.0

    def compute_doc_state(self, doc):
        return self.bow_to_clusters(self.tokens_to_bow_vector(doc.text.split()))

    @lru_cache(maxsize=5000)
    def get_entity_context_vec(self, entity):
        return self.bow_to_clusters(self.counts_to_bow(self.context_model[entity]))
    
    def distance(self, query, entity):
        a = query
        b = entity

        if len(b) < len(a):
            a, b = b, a

        limit = 3
        if len(a) > limit:
            a = sorted(a,key=len,reverse=True)[:limit]

        sims = []
        for ca in a:
            max_sim = None
            max_i = None
            for i, cb in enumerate(b):
                sim = numpy.dot(ca, cb)
                if sim > max_sim:
                    max_sim, max_i = sim, i
            sims += [max_sim]
            del b[max_i]

        return numpy.mean(sims)

    @lru_cache(maxsize=None)
    def term_vector(self, t, weight):
        return self.wordvec_model.word_to_vec(t)*weight

    def iter_word_reps(self, bow):
        for t, weight in bow.iteritems():
            yield self.term_vector(t, weight)

    def bow_to_clusters(self, bow):
        return self.cluster(self.iter_word_reps(bow)) if bow else []

    def cluster(self, wordreps):
        clusters = []

        for v in wordreps:
            if clusters:
                n = float(len(clusters))
                max_i, max_sim = None, None
                for i, c in enumerate(clusters):
                    sim = numpy.dot(c / numpy.linalg.norm(c), v / numpy.linalg.norm(v))
                    if sim > max_sim:
                        max_i, max_sim = i, sim
                
                r = (1/(1+n))
                if max_sim > r or random.random() > r:
                    clusters[i] += v
                else:
                    clusters.append(v)
            else:
                clusters.append(v)
        
        # normalise cluster vectors
        for i, c in enumerate(clusters):
            clusters[i] = c / numpy.linalg.norm(c)

        return clusters
    
    @classmethod
    def add_arguments(cls, p):
        super(CRPClusteredDBoWContext, cls).add_arguments(p)
        p.add_argument('wordvec_model_path', metavar='WORD_VECTOR_MODEL')
        return p

