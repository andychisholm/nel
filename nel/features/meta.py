#!/usr/bin/env python
import operator
import cPickle as pickle
import numpy as np
import math
import json
import marshal

from pymongo import MongoClient
from collections import defaultdict
from functools32 import lru_cache
from scipy.sparse import csc_matrix

from .feature import Feature
from .mapping import FEATURE_MAPPERS
from ..model import model
from ..model.model import EntityOccurrence

import logging
log = logging.getLogger()

class ClassifierFeature(Feature):
    """ Computes a feature score based on the output of a classifier over a set of features. """
    def __init__(self, classifier):
        log.info('Loading classifier (%s)...', classifier)
        self._id = classifier
        self.classifier = model.Classifier(classifier)

    def compute_doc_state(self, doc):
        doc = self.classifier.mapper(doc) 

    def predict(self, fv):
        raise NotImplementedError # returns numerical prediction given a vector of features

    def compute(self, doc, chain, candidate, state):
        return self.predict(candidate.fv)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('classifier', metavar='CLASSIFIER')
        p.set_defaults(featurecls=cls)
        return p

@Feature.Extractable
class ClassifierScore(ClassifierFeature):
    """ Computes a feature score based on the output of the classifier decision function over a set of features. """
    @property
    def id(self):
        return 'ClassifierScore[%s]' % self._id

    def predict(self, fv):
        return float(self.classifier.model.decision_function(fv))

@Feature.Extractable
class ClassifierProbability(ClassifierFeature):
    """ Computes a feature score based on the output probability of a classifier over a set of features. """
    @property
    def id(self):
        return 'ClassifierProbability[%s]' % self._id

    def predict(self, fv):
        return self.classifier.model.predict_proba(fv)[0][1]

class CoherenceBase(Feature):
    def __init__(self, ranker):
        self.ranking_feature = ranker
        self.depth = 3

    @property
    def id(self):
        return self.__class__.__name__ + '[' + self.ranking_feature + ']'
 
    def compute_doc_state(self, doc):
        rankings = {}
        for chain in doc.chains:
            rankings[chain] = sorted(chain.candidates,key=lambda c: c.features[self.ranking_feature],reverse=True)[:self.depth]
        
        return rankings

    def compute(self, doc, chain, candidate, rankings): 
        scores = []
        
        for ch in doc.chains:
            if ch != chain:
                if rankings[ch]:
                    scores.append(max(self.score(candidate.id, c.id) for c in rankings[ch]))
                else:
                    scores.append(-20)
        if scores:
            return sum(scores)/len(scores)
        else:
            return -20*30
    def score(self, a, b):
        raise NotImplementedError

@Feature.Extractable
class MeanConditionalProbability(CoherenceBase):
    """ Feature which measures relatedness between candidates in the document """
    def __init__(self, ranker, occurrence_model_path):
        super(MeanConditionalProbability, self).__init__(ranker)
        self.occurrence_model = EntityOccurrence.read(occurrence_model_path)

    @lru_cache(maxsize=100000)
    def score(self, a, b):
        occ_a = self.occurrence_model.occurrences(a)
        occ_b = self.occurrence_model.occurrences(b)
        
        if len(occ_b) < len(occ_a):
            occ_a, occ_b = occ_b, occ_a

        inter = .0
        union = .0
        for e in occ_a:
            union += 1
            if e in occ_b:
                inter += 1
        union += len(occ_b) - inter

        try:
            return math.log(inter/union)
        except (ZeroDivisionError, ValueError):
            return -20.0
    
    @classmethod
    def add_arguments(cls, p):
        p.add_argument('ranker', metavar='RANKING_FEATURE_ID')
        p.add_argument('occurrence_model_path', metavar='OCCURRENCE_MODEL')
        p.set_defaults(featurecls=cls)
        return p

"""
from gensim.models.doc2vec import Doc2Vec

@Feature.Extractable
class EmbeddingCoherence(Feature):
    " Computes a relatedness feature score based on coherence between entities in the candidate set."
    def __init__(self, cooccurence_model_path, ranker):
        model_path = '/data0/linking/models/wikipedia_conll.embeddings.model'
        self.embedding_model = Doc2Vec.load(model_path, mmap='r')
        self.redirects = marshal.load(open('/data0/linking/models/wikipedia.redirect.model', 'rb'))
        self.score_model = pickle.load(open(score_model_path, 'rb'))

    @lru_cache(maxsize=None)
    def get_top_scores(self, doc_id, mention_begin, mention_end, n):
        top_scores = sorted(self.score_model[doc_id][(mention_begin, mention_end)].iteritems(), key=operator.itemgetter(1), reverse=True)[:n]
        return [(s/top_scores[0][1], c) for c, s in top_scores]

    @lru_cache(maxsize=None)
    def similarity(self, a, b):
        a = '~' + self.redirects.get(a, a)
        b = '~' + self.redirects.get(b, b)

        if a not in self.embedding_model.vocab or b not in self.embedding_model.vocab:
            return 2.0

        return 1.0 - self.embedding_model.similarity(a, b)

    def compute(self, doc, chain, candidate, state): 
        scores = []
        for m in doc.mentions:
            if m != mention:
                scores.append( min(self.similarity(candidate, e) for _, e in self.get_top_scores(doc.id, mention.begin, mention.end, 3)) )
        
        if not scores:
            return 2.0
        return sum(scores) / len(scores)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('ranker', metavar='RANKING_FEATURE_ID')
        p.add_argument('cooccurence_model_path', metavar='COCCURRENCE_MODEL_PATH')
        p.set_defaults(featurecls=cls)
        return p

@Feature.Extractable
class PageRankCoherence(ConditionalCoherence):
    " Computes candidate coherence based on pagerank of candidates over the document mention-candidate graph "
    
    @staticmethod
    def pageRank(G, s = .85, maxerr = .001, maxiter=1000):
        " adapted from: https://gist.github.com/diogojc/1338222 " 
        n = G.shape[0]

        # transform G into markov matrix M
        M = csc_matrix(G,dtype=np.float)
        rsums = np.array(M.sum(1))[:,0]
        ri, ci = M.nonzero()
        M.data /= rsums[ri]

        # bool array of sink states
        sink = rsums==0

        # Compute pagerank r until we converge
        ro, r = np.zeros(n), np.ones(n)

        iterations = 0 
        while iterations < maxiter and np.sum(np.abs(r-ro)) > maxerr:
            ro = r.copy()

            # calculate each pagerank at a time
            for i in xrange(0,n):
                # inlinks of state i
                Ii = np.array(M[:,i].todense())[:,0]
                # account for sink states
                Si = sink / float(n)
                # account for teleportation to state i
                Ti = np.ones(n) / float(n)

                r[i] = ro.dot( Ii*s + Si*s + Ti*(1-s) )

            iterations += 1

        return r

        # return normalized pagerank
        #return r/sum(r)       

    def _cp(self, a, b):
        occ_a = self.cooccurence_model.occurrences(a)
        occ_b = self.cooccurence_model.occurrences(b)
        num_common = len(occ_a.intersection(occ_b))
        num_union = len(occ_b)

        return 0.0 if num_union == 0 else float(num_common) / float(num_union)
    
    @lru_cache(maxsize=1000000)
    def ref(self, a, b):
        return b in self.cooccurence_model.occurrences(a) or a in self.cooccurence_model.occurrences(b)

    @lru_cache(maxsize=1000000)
    def num_common(self, a, b):
        return len(self.cooccurence_model.occurrences(a).intersection(self.cooccurence_model.occurrences(b)))

    def compute(self, doc, chain, candidate, state):
        # bit of a hack to ensure we only compute pageranks once per document
        # despite features being called for on a per-candidate basis
        if not hasattr(doc, 'page_ranks'):
            doc.page_ranks = defaultdict(lambda: defaultdict(float))
            
            mention_candidates = [(m, c) for m in doc.mentions for c in self.get_top_scores(doc, m, 3)]
            n = len(mention_candidates)

            #log.debug('[%s] Computing pagerank over %i mention-candidates...' % (doc.id, n))
            #start_time = time()

            M = np.zeros((n, n))
            for i, (mi, (si, ci)) in enumerate(mention_candidates):
                for j, (mj, (sj, cj)) in enumerate(mention_candidates):
                    if i != j and mi.text.lower() != mj.text.lower():
                        M[i,j] = 20.0 + self.conditional_probability(ci, cj)
                        #M[i,j] = 1.0 if self.ref(ci, cj) or self.num_common(ci, cj) > 0 else 0.0
            
            # normalisation of the link-weight matrix?
            c_sums = M.sum(axis=0)
            c_sums[c_sums==0] = 1. # ensure we don't introduce NaN's for all-zero columns
            M = M / c_sums

            ranks = self.pageRank(M, s = .85, maxerr=0.01)
            
            #log.debug('[%s] Completed %i mention-candidates in %i seconds...' % (doc.id, n, int(time() - start_time)))
            
            for i, (m, (s, c)) in enumerate(mention_candidates):
                pr = ranks[i]
                # todo: normalisation of pagerank scores amungst a mention's candidate set
                #       if we don't do this, candidate scores may be diluted by the size of the mention-candidate graph
                #       alternatively, we could remove normalisation from the end of the pagerank calculation above

                log.debug('%s: S = %.4f, PR = %.4f' % (c, s, pr))
                doc.page_ranks[m][c] = pr

        return doc.page_ranks[mention][candidate]
"""
