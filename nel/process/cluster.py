from nel.doc import Candidate
from collections import defaultdict

class IterativeClusterer(object):
    def get_cluster_key_for_chain(self, clusters_by_key, chain):
        raise NotImplementedError

    def __call__(self, docs):
        clusters_by_key = defaultdict(list)

        for d in docs:
            for c in d.chains:
                if c.resolution == None and c.mentions:
                    clusters_by_key[self.get_cluster_key_for_chain(clusters_by_key, c)].append(c)

        for i, (key, chains) in enumerate(clusters_by_key.iteritems()):
            for c in chains:
                c.resolution = Candidate('null/'+str(i))

        return docs

class NameClusterer(IterativeClusterer):
    def get_cluster_key_for_chain(self, clusters_by_key, chain):
        return sorted(chain.mentions, key=len, reverse=True)[0].text.lower().replace(' ', '_')

name = NameClusterer

from ..util import get_from_module
def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'clusterer', instantiate=True, kwargs=kwargs)
