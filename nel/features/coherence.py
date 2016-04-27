import random
import numpy

from .feature import Feature
from ..model.disambiguation import EntityEmbeddings

from scipy.spatial.distance import cosine
from scipy.stats import gmean

from nel import logging
log = logging.getLogger()

@Feature.Extractable
class EmbeddingCoherence(Feature):
    """ Coherence measure based on minimum distances between the highest ranked entities in the document """
    def __init__(self, embedding_model_tag, ranking_feature):
        self.tag = embedding_model_tag
        self.ranking_feature = ranking_feature
        self.em = EntityEmbeddings(self.tag)
 
        # threshold on how many candidates of other 
        # chains in the doc are considered when
        # calculating minimum distance to a candidate
        self.coherence_depth = 2

        # threshold on the minimum rank of candidates
        # for which coherence will be computed
        # rerank depth must be >= coherence depth
        self.rerank_depth = 5

        # maximum number of chains with which coherence is compared
        # if len(chains) is greater then chains are randomly subsampled
        # lowering this limit improves worst case performance for huge documents
        self.max_coherent_chains = 250

    @property
    def id(self):
        return self.__class__.__name__ + '[' + self.tag + '][' + self.ranking_feature + ']'

    def distance(self, query, entity):
        if query == None or entity == None:
            return 2.
        return cosine(query, entity)

    def compute_doc_state(self, doc):
        # fetch entity embedding for the set of candidates being considered
        candidates = set()
        for chain in doc.chains:
            for c in sorted(chain.candidates, key=lambda c: c.features[self.ranking_feature], reverse=True)[:self.rerank_depth]:
                    candidates.add(c.id)
        candidate_embeddings = self.em.get_embeddings(candidates)

        # ensure we only compute pairwise distance once
        distance_cache = {}
        def candidate_distance(a, b):
            if a < b:
                a, b = b, a
            key = (a,b)
            if key not in distance_cache:
                distance_cache[key] = self.distance(candidate_embeddings[a], candidate_embeddings[b])
            return distance_cache[key]

        # if the document is too large, only consider coherence with a random subset
        coherent_chains = doc.chains if len(doc.chains) < self.max_coherent_chains else random.sample(doc.chains, self.max_coherent_chains)

        # precompute the top candidates for each chain
        rc_by_chain = {}
        for chain in coherent_chains:
            rc_by_chain[chain] = [c.id for c in sorted(chain.candidates, key=lambda c: c.features[self.ranking_feature], reverse=True)][:self.coherence_depth+1]

        state = {}
        for c in candidates:
            dists = []
            for chain in coherent_chains:
                top = [ci for ci in rc_by_chain[chain] if ci != c][:self.coherence_depth]
                if not top:
                    continue
                dists.append(min(candidate_distance(c, tc) for tc in top if tc != c))
            if dists:
                state[c] = gmean(dists)

        return state

    def compute(self, doc, chain, candidate, state):
        return state.get(candidate.id, 2.)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('embedding_model_tag', metavar='EMBEDDING_MODEL')
        p.add_argument('ranking_feature', metavar='RANKING_FEATURE')
        p.set_defaults(featurecls=cls)
        return p

class CoherenceBase(Feature):
    def __init__(self, ranker):
        self.ranking_feature = ranker
        self.depth = 2

    @property
    def id(self):
        return self.__class__.__name__ + '[' + self.ranking_feature + ']'

    def default_coherence(self):
        return -20
 
    def compute_doc_state(self, doc):
        rankings = {}
        for chain in doc.chains:
            rankings[chain] = sorted(chain.candidates,key=lambda c: c.features[self.ranking_feature],reverse=True)[:self.depth]
        
        return rankings

    def compute(self, doc, chain, candidate, rankings): 
        scores = []
        
        if candidate not in rankings[chain]:
            return self.default_coherence()*30

        for ch in doc.chains:
            if ch != chain:
                if rankings[ch]:
                    scores.append(max(self.score(candidate.id, c.id) for c in rankings[ch]))
                else:
                    scores.append(-20)
        if scores:
            return sum(scores)/len(scores)
        else:
            return self.default_coherence()*30

    def score(self, a, b):
        raise NotImplementedError
