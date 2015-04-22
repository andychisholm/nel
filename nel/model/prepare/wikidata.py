import gzip
import ujson as json

from time import time
from ..model import Redirects
from ..data import Store

import logging
log = logging.getLogger()

def iter_wikidata_items(dump_path):
    # used to estimate runtime, given one json item per line
    # we can guess this via: zcat dump.json.gz|wc -l
    EST_DESC_TOTAL = 17272576

    start_time = time()
    with gzip.open(dump_path, 'r') as gzf:
        for i, line in enumerate(gzf):
            if i == 50000 or i % 500000 == 0:
                completed = i / float(EST_DESC_TOTAL)
                elapsed = time() - start_time
                eta = 0 if i == 0 else ((elapsed / completed)-elapsed) / 60

                log.debug('Processed %i wikidata items, ~%.1f%%, eta=%.2fm ...', i, completed*100, eta)
            
            try:
                yield json.loads(line[:-2])
            except ValueError:
                # todo: verify this only occurs on first/last lines of file
                continue

def normalise_wikipedia_title(title):
    return title.replace(' ', '_')

class BuildWikidataRedirects(object):
    """ Extracts a mapping from wikidata to wikipedia entities """
    def __init__(self, wikidata_dump_path, wikipedia_redirect_model_tag, model_tag):
        self.wikidata_dump_path = wikidata_dump_path
        self.wikipedia_redirect_model_tag = wikipedia_redirect_model_tag
        self.model_tag = model_tag
        
    def __call__(self):
        log.info('Building redirect model from: %s' % self.wikidata_dump_path)
        Redirects(self.model_tag).create(self.iter_mappings())
        log.info('Done.')

    def iter_mappings(self):
        log.info('Prefetching wikipedia redirects...')
        wikipedia_redirects = Redirects(self.wikipedia_redirect_model_tag).dict()

        for obj in iter_wikidata_items(self.wikidata_dump_path):
            if obj['id'][0] == 'Q':
                try:
                    wikidata_id = obj['id'][1:]
                    entity = normalise_wikipedia_title(obj['sitelinks']['enwiki']['title'])
                    entity = wikipedia_redirects.get(entity, entity)
                    yield wikidata_id, entity
                except KeyError:
                    continue # skip entities missing english sitelinks

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('wikidata_dump_path', metavar='WIKIDATA_DUMP_PATH')
        p.add_argument('--wikipedia_redirect_model_tag', required=False, default='wikipedia', metavar='WIKIPEDIA_REDIRECT_MODEL_TAG')
        p.add_argument('--model_tag', required=False, default='wikidata', metavar='WIKIDATA_REDIRECT_MODEL_TAG')
        p.set_defaults(cls=cls)
        return p

class BuildWikidataTriples(object):
    """ Extracts an relation triples from a wikidata dump """
    def __init__(self, wikidata_dump_path):
        self.wikidata_dump_path = wikidata_dump_path

    def __call__(self):
        for i, (sub, pred, obj) in enumerate(iter_triples()):
            # todo: something!
            if i % 250000 == 0:
                log.info('Processed %i relation triples...', i)

    def iter_triples(self):
        for obj in iter_wikidata_items(self.wikidata_dump_path):
            if obj['id'][0] != 'Q':
                continue
            for pid, statements in obj.get('claims', {}).iteritems():
                pid = int(pid[1:])

                for statement in statements:
                    datatype = statement['mainsnak'].get('datatype', None)

                    if datatype == 'wikibase-item':
                        tid = statement['mainsnak']['datavalue']['value']['numeric-id']
                        yield (int(obj['id'][1:]), pid, int(tid))
    @classmethod
    def add_arguments(cls, p):
        p.add_argument('wikidata_dump_path', metavar='WIKIDATA_DUMP_PATH')
        p.set_defaults(cls=cls)
        return p

R_PD_MATCH=')'
L_PD_MATCH=' ('
class BuildWikidataEntityDescriptions(object):
    """ Extracts an entity->short_description store from a wikidata dump """    
    def __init__(self, wikidata_dump_path):
        self.wikidata_dump_path = wikidata_dump_path

    def __call__(self):
        store = Store.Get('models:descriptions')
        with store.batched_inserter(250000) as s:
            for obj in iter_wikidata_items(self.wikidata_dump_path):
                try:
                    entity = obj['sitelinks']['enwiki']['title']
                except KeyError:
                    continue # skip entities missing english sitelinks

                description = obj.get('descriptions',{}).get('en',{}).get('value', '')
                label = obj.get('labels',{}).get('en',{}).get('value', None)

                # try to populate missing descriptions using parenthetical disambiguations
                # populate missing entity labels based on the entity id
                if not label or not description:
                    if entity.endswith(R_PD_MATCH):
                        pd_start = entity.rfind(L_PD_MATCH)
                        if pd_start != -1:
                            if not label:
                                label = entity[:pd_start]
                            if not description:
                                description = entity[pd_start+len(L_PD_MATCH):-1]
                    if not label:
                        label = entity

                s.save({
                    '_id': normalise_wikipedia_title(entity),
                    'label': label,
                    'description': description
                })

        log.info('Done')

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('wikidata_dump_path', metavar='WIKIDATA_DUMP_PATH')
        p.set_defaults(cls=cls)
        return p
