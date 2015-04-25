import gzip
import ujson as json
from time import time
from collections import defaultdict

from ..model import Entities, Redirects

import logging
log = logging.getLogger()

def iter_wikidata_items(dump_path):
    # used to estimate runtime, given one json item per line
    # we can guess this via: zcat dump.json.gz|wc -l
    EST_DESC_TOTAL = 17272576

    start_time = time()
    with gzip.open(dump_path, 'r') as gzf:
        for i, line in enumerate(gzf):
            if i == 0 and line == '[' or line == ']':
                continue
            if i == 50000 or i % 500000 == 0:
                completed = i / float(EST_DESC_TOTAL)
                elapsed = time() - start_time
                eta = 0 if i == 0 else ((elapsed / completed)-elapsed) / 60

                log.debug('Processed %i wikidata items, ~%.1f%%, eta=%.2fm ...', i, completed*100, eta)
            
            try:
                yield json.loads(line.rstrip(',\n'))
            except ValueError:
                if line.strip() == '[' or line.strip() == ']':
                    continue
                raise

def iter_wikidata_relations(dump_path):
    for item in iter_wikidata_items(dump_path):
        # only consider relations over entities
        if item['id'][0] == 'Q':
            for r in iter_relations_for_item(item):
                yield r

def iter_relations_for_item(item):
    for pid, statements in item.get('claims', {}).iteritems():
        pid = int(pid[1:])

        for statement in statements:
            datatype = statement['mainsnak'].get('datatype', None)

            if datatype == 'wikibase-item':
                tid = statement['mainsnak']['datavalue']['value']['numeric-id']
                yield (int(item['id'][1:]), pid, int(tid))

R_PD_MATCH=')'
L_PD_MATCH=' ('
def get_entity_for_item(item):
    try:
        # todo: we could support extraction for non-wikipedia kbs here
        entity = item['sitelinks']['enwiki']['title']
    except KeyError:
        return None # skip out-of-KB wikidata entities

    description = item.get('descriptions',{}).get('en',{}).get('value', '')
    label = item.get('labels',{}).get('en',{}).get('value', None)

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

    return normalise_wikipedia_title(entity), label, description

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

class BuildWikidataEntitySet(object):
    """ Builds an entity set over a wikidata subgraphs """
    ENTITY_ITEM_ID = 35120
    INSTANCE_OF_PID = 31
    SUBCLASS_OF_PID = 279

    def __init__(self, wikidata_dump_path, model_tag, include, exclude):
        self.wikidata_dump_path = wikidata_dump_path
        self.model_tag = model_tag
        self.included_nodes = set(include or [])
        self.excluded_nodes = set(exclude or [])

    def __call__(self):
        if not self.included_nodes:
            log.warn("No nodes specified for inclusion, all objects derived from Entity will be included.")
            self.included_nodes = [self.ENTITY_ITEM_ID]

        entities = {}
        relation_graph = defaultdict(list)

        entity_count = 0
        relation_count = 0
        for item in iter_wikidata_items(self.wikidata_dump_path):
            if item['id'][0] == 'Q':
                wikidata_id = int(item['id'][1:])

                # extract relations
                for sub, pid, obj in iter_relations_for_item(item):
                    if pid == self.INSTANCE_OF_PID or pid == self.SUBCLASS_OF_PID:
                        relation_graph[obj].append(sub)

                    relation_count += 1
                    if relation_count % 1000000 == 0:
                        log.info('Processed %i relation triples...', relation_count)

                # extract entity info if this is a wikipedia entity
                entity = get_entity_for_item(item)
                if entity:
                    entity_count += 1
                    entities[wikidata_id] = entity
                    if entity_count % 250000 == 0:
                        log.info('Processed %i entities...', entity_count)

        log.info('Filtering %i entities...', entity_count)
        included_entities = set()
        for node in self.included_nodes:
            node_count = 0
            for eid in self.iter_leaves(relation_graph, node, self.excluded_nodes):
                if eid in entities:
                    included_entities.add(eid)
                    node_count += 1
            log.debug("Total = %i entities after merging %i derived from Q%i", len(included_entities), node_count, node)

        # there seem to be a few minors errors with wikidata sitelinks where multiple wikidata
        # items map to the same wikipedia entity, e.g. 671315 and 19371531 both map to 'Helmeringhausen'
        # no nice way to resolve these, so just remap the dict on wikipedia id to blow away the dupes
        entities = {e[0]:e for k,e in entities.iteritems() if k in included_entities}
        if len(entities) != len(included_entities):
            log.warn("Filtered %i entities mapping to more than one wikidata item...", len(included_entities) - len(entities))

        entity_model = Entities(self.model_tag)
        entity_model.create(entities.itervalues())

    def iter_leaves(self, graph, root, excluded = None):
        # tracks visited nodes; excludes nodes from traversal if pre-populated
        excluded = set() if excluded == None else excluded
        pending = list(graph[root])
        excluded = excluded.union(pending)

        # not quite as pretty, but faster than the recursive version
        while pending:
            node = pending.pop()
            excluded.add(node)

            children = graph.get(node, None)
            if children:
                pending += [n for n in children if n not in excluded]
            else:
                yield node

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('wikidata_dump_path', metavar='WIKIDATA_DUMP_PATH')
        p.add_argument('--model_tag', metavar='ENTITIES_MODEL_TAG', required=False, default='wikipedia')
        p.add_argument('--include', metavar='INCLUDE_NODE', type=int, action='append')
        p.add_argument('--exclude', metavar='EXCLUDE_NODE', type=int, action='append')
        p.set_defaults(cls=cls)
        return p
