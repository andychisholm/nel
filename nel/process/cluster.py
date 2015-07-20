from collections import defaultdict

class IterativeClusterer(object):
    def get_cluster_key_for_chain(self, clusters_by_key, chain):
        raise NotImplementedError

    def __call__(self, docs):
        clusters_by_key = defaultdict(list)
        for d in docs:
            for c in d.chains:
                clusters_by_key[self.get_cluster_key_for_chain(clusters_by_key, c)].append(c)
        return clusters_by_key.values()

class NameClusterer(IterativeClusterer):
    def get_cluster_key_for_chain(self, clusters_by_key, chain):
        return sorted(chain.mentions, key=len, reverse=True)[0].text.lower()

name = NameClusterer

from ..util import get_from_module
def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'clusterer', instantiate=True, kwargs=kwargs)
