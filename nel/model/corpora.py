from .data import ObjectStore

from nel import logging
log = logging.getLogger()

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
