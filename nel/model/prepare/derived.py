#!/usr/bin/env python
import math
import re
import logging
import marshal

from collections import defaultdict,Counter
from schwa import dr
from .. import model

from ..model import Redirects
from ..model import WordVectors
from ..model import EntityMentionContexts
from ..model import mmdict
from ..model import Links
from ..model import EntityOccurrence, EntityCooccurrence
from ..model import Candidates

from .util import trim_subsection_link

log = logging.getLogger()

class BuildEntitySet(object):
    "Build redirect model from wikipedia-redirect tsv."
    def __init__(self, page_model_path, entity_set_model_path):
        self.page_model_path = page_model_path
        self.entity_set_model_path = entity_set_model_path

    def __call__(self):
        log.info('Building entity set for corpus: %s', self.page_model_path)

        class Doc(dr.Doc):
            name = dr.Field()

        entities = set()

        with open(self.page_model_path,'r')  as f:
            reader = dr.Reader(f, Doc.schema())
            for i, doc in enumerate(reader):
                if i % 500000 == 0:
                    log.debug('Processed %i documents...', i)
                entities.add(doc.name)

        log.info('Writing entity set model to file: %s', self.entity_set_model_path)
        with open(self.entity_set_model_path, 'wb') as f:
            marshal.dump(entities, f)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('page_model_path', metavar='PAGE_MODEL_MODEL')
        p.add_argument('entity_set_model_path', metavar='ENTITY_SET_MODEL_MODEL')
        p.set_defaults(cls=cls)
        return p

class BuildLinkModels(object):
    "Build link derived models from a docrep corpus."
    def __init__(self, page_model_path, entity_set_model_path, model_tag):
        self.page_model_path = page_model_path
        self.entity_set_model_path = entity_set_model_path
        self.model_tag = model_tag

    def __call__(self):
        log.info('Building page link derived models from: %s', self.page_model_path)

        class Link(dr.Ann):
            anchor = dr.Field()
            target = dr.Field()

        class Doc(dr.Doc):
            name = dr.Field()
            links = dr.Store(Link)

        log.info('Loading redirects...')
        redirects = Redirects().dict()
        entity_counts = defaultdict(int)
        name_entity_counts = defaultdict(lambda:defaultdict(int))
        # occurrence = EntityOccurrence() # todo: occurrence model datastore backend

        log.info('Loading entity set...')
        entity_set = marshal.load(open(self.entity_set_model_path, 'rb'))

        with open(self.page_model_path,'r')  as f:
            reader = dr.Reader(f, Doc.schema())
            for i, doc in enumerate(reader):
                if i % 100000 == 0:
                    log.info('Processed %i documents...', i)
                
                for link in doc.links:
                    # we may want to ignore subsection links when generating name models
                    # sometimes a page has sections which refer to subsidiary entities
                    # links to these may dilute the name posterior for the main entity
                    # for now we just add everything to keep it simple
                    target = trim_subsection_link(link.target)
                    target = redirects.get(target, target)
                    target = trim_subsection_link(link.target)

                    #occurrence.add(target, doc.name)

                    # ignore out-of-kb links in entity and name probability models
                    if target in entity_set:
                        entity_counts[target] += 1
                        name_entity_counts[self.normalise_name(link.anchor)][target] += 1
        
        ep_model = model.EntityPrior(self.model_tag)
        ep_model.create(entity_counts.iteritems())
        entity_counts = None
        nep_model = model.NameProbability(self.model_tag)
        nep_model.create(name_entity_counts.iteritems())
        nep_model = None
        
        log.info('Done')

    def normalise_name(self, name):
        return name.lower()

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('page_model_path', metavar='PAGE_MODEL_MODEL')
        p.add_argument('entity_set_model_path', metavar='ENTITY_SET_MODEL_PATH')
        p.add_argument('model_tag', metavar='MODEL_TAG')
        p.set_defaults(cls=cls)
        return p

class BuildTitleRedirectNames(object):
    "Builds alias model using the title of an entity page (e.g. Some_Entity) and the title of redirects."
    def __init__(self, title_model_path, redirect_model_path, alias_model_path):
        self.title_model_path = title_model_path
        self.redirect_model_path = redirect_model_path
        self.alias_model_path = alias_model_path
        self.underscores_re = re.compile('_+')

    def convert_title_to_name(self, entity):
        # strip parts after a comma
        comma_idx = entity.find(',_')
        if comma_idx > 0:
            entity = entity[:comma_idx]

        # strip components in brackets
        lb_idx = entity.rfind('_(')
        if lb_idx > 0:
            rb_idx = entity.find(')', lb_idx)
            if rb_idx > 0:
                entity = entity[:lb_idx] + entity[rb_idx+1:]
        
        # replace underscores with spaces
        return self.underscores_re.sub(' ', entity)

    def __call__(self):
        log.info('Loading entity titles: %s' % self.title_model_path)
        titles = marshal.load(open(self.title_model_path, 'rb'))

        log.info('Loading entity redirects: %s' % self.redirect_model_path)
        redirects = Redirects.read(self.redirect_model_path).get_redirects_by_entity()

        log.info('Processing entity titles...')
        aliases = {}
        alias_count = 0
        for entity in titles:
            entity_titles = set(redirects.get(entity, []))
            entity_titles.add(entity)
            
            entity_aliases = set(self.convert_title_to_name(t) for t in entity_titles)
            alias_count += len(entity_aliases)
            aliases[entity] = list(entity_aliases)

        log.info('Writing entity alias model (%i entities, %i names): %s' % (len(aliases), alias_count, self.alias_model_path))
        marshal.dump(aliases, open(self.alias_model_path,'wb'))

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('title_model_path', metavar='IN_TITLE_MODEL_FILE')
        p.add_argument('redirect_model_path', metavar='IN_REDIRECT_MODEL_FILE')
        p.add_argument('alias_model_path', metavar='OUT_ALIAS_MODEL_FILE')
        p.set_defaults(cls=cls)
        return p

def get_mention_term_counts(emw):
    entity, mentions, window = emw
    rhs_len = None if window == None else window / 2
    lhs_len = None if window == None else window - rhs_len

    global ngram_vocab

    counts = Counter()
    for _, l, n, r in mentions:
        lhs, name, rhs = tokenize(l), tokenize(n), tokenize(r)

        if window == None: tokens = lhs + name + rhs
        else:              tokens = lhs[-lhs_len:] + name + rhs[:rhs_len]

        counts.update(ngrams(tokens, 3, ngram_vocab))

    return (entity, counts)

def iter_common_lemma_names():
    for synset in wordnet.all_synsets():
        for name in synset.lemma_names:
            if name.islower():
                yield name

class BuildCandidateModel(object):
    "Builds an alias model from a name probability model."
    def __init__(self, alias_model_path, name_model_path, title_model_path):
        self.alias_model_path = alias_model_path
        self.name_model_path = name_model_path
        
        self.title_model_path = title_model_path
        self.underscores_re = re.compile('_+')

        log.info('Loading redirect mapping...')
        self.redirects = Redirects().dict()
 
    def convert_title_to_name(self, entity):
        # strip parts after a comma
        comma_idx = entity.find(',_')
        if comma_idx > 0:
            entity = entity[:comma_idx]

        # strip components in brackets
        lb_idx = entity.rfind('_(')
        if lb_idx > 0:
            rb_idx = entity.find(')', lb_idx)
            if rb_idx > 0:
                entity = entity[:lb_idx] + entity[rb_idx+1:]
        
        # replace underscores with spaces
        return self.underscores_re.sub(' ', entity)
    
    def iter_entity_aliases(self):
        log.info('Loading aliases: %s ...', self.alias_model_path)
        alias_model = marshal.load(open(self.alias_model_path,'rb')) 
        log.info('Enumerating aliases for %i entities...' % len(alias_model))
        for entity, names in alias_model.iteritems():
            entity = self.redirects.get(entity, entity)
            for name in names:
                yield entity, name
        alias_model = None

        log.info('Enumerating redirects...')
        for source, target in self.redirects.iteritems():
            yield target, self.convert_title_to_name(source)

        log.info('Loading entity ids: %s ...', self.title_model_path)
        titles = marshal.load(open(self.title_model_path,'rb'))
        for entity in titles:
            entity = self.redirects.get(entity, entity)
            yield entity, self.convert_title_to_name(entity)
        titles = None

        name_model = Name()
        name_model.read(self.name_model_path)
        name_model = name_model.get_entities_by_name()
        for name, entities in name_model.iteritems():
            for entity in entities:
                yield self.redirects.get(entity, entity), name
        name_model = None

    def __call__(self):
        log.info('Building candidate model...')
        Candidates().create(self.iter_entity_aliases())

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('alias_model_path', metavar='ALIAS_MODEL_PATH')
        p.add_argument('name_model_path', metavar='NAME_MODEL_PATH')
        p.add_argument('title_model_path', metavar='ENTITY_ID_MODEL_PATH')
        p.set_defaults(cls=cls)
        return p

class BuildEntityCooccurrenceFromOccurrence(object):
    "Builds model of entity conditional probability from occurrence model."
    def __init__(self, occurrence_model_path, probability_model_path):
        self.occurrence_model_path = occurrence_model_path
        self.out_path = probability_model_path

    def __call__(self):
        log.info('Building entity coocurrence statistics "from entity occurrence model...')
        occurrence_model = EntityOccurrence.read(self.occurrence_model_path)

        log.info('Computing occurrence counts...')
        occurrence_counts = {k:len(e) for k, e in occurrence_model.d.iteritems()}

        log.info('Inverting occurrence model..')
        entities_by_occurrence = defaultdict(set)
        for e, occurrences in occurrence_model.d.iteritems():
            for o in occurrences:
                entities_by_occurrence[o].add(e)

        occurrence_model = None

        log.info('Computing cooccurrence counts over %i pages...' % len(entities_by_occurrence))
        cooccurrence_counts = {}
        for i, loc in enumerate(entities_by_occurrence.keys()):
            entities = entities_by_occurrence[loc]

            if i % 500000 == 0:
                log.info('Processed %i pages... %i pairs', i, len(cooccurrence_counts))
            for a in entities:
                for b in entities:
                    if a != b:
                        if a > b:
                            a, b = b, a
                        if a not in cooccurrence_counts:
                            cooccurrence_counts[a] = {b:1}
                        else:
                            if b not in cooccurrence_counts[a]:
                                cooccurrence_counts[a][b] = 1
                            else:
                                cooccurrence_counts[a][b] += 1

        log.info('Building EntityCooccurrence model...')
        ec = EntityCooccurrence(cooccurrence_counts, occurrence_counts)
        ec.write(self.out_path)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('occurrence_model_path', metavar='OCCURRENCE_MODEL')
        p.add_argument('probability_model_path', metavar='PROBABILITY_MODEL')
        p.set_defaults(cls=cls)
        return p

class BuildOccurrenceFromLinks(object):
    "Builds link cooccurrence model from outlink model."
    def __init__(self, link_model_path, redirect_model_path, occurrence_model_path):
        self.link_model_path = link_model_path
        self.out_path = occurrence_model_path

        log.info('Loading redirect model: %s ...', redirect_model_path)
        self.redirect_model = marshal.load(open(redirect_model_path, 'rb'))

    def __call__(self):
        log.info('Building entity occurrence model from outlinks...')
        occurrence_model = EntityOccurrence()
        link_model = Links.read(self.link_model_path)

        for page, links in link_model.iteritems():
            for e in links:
                hidx = e.rfind('#')
                if hidx != -1:
                    e = e[:hidx]
                e = self.redirect_model.get(e, e)
                occurrence_model.add(e, page)

        occurrence_model.write(self.out_path)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('link_model_path', metavar='LINK_MODEL')
        p.add_argument('redirect_model_path', metavar='REDIRECT_MODEL_PATH')
        p.add_argument('occurrence_model_path', metavar='OCCURRENCE_MODEL')
        p.set_defaults(cls=cls)
        return p

class BuildOccurrenceFromMentions(object):
    "Builds link cooccurrence model from entity mention contexts."
    def __init__(self, mention_model_path, occurrence_model_path):
        self.mention_model_path = mention_model_path
        self.out_path = occurrence_model_path

    def __call__(self):
        log.info('Building entity occurrence model from mention contexts...')
        occurence_model = EntityOccurrence()

        mention_iter = EntityMentionContexts.iter_entity_mentions_from_path(self.mention_model_path)
        for i, (e, mentions) in enumerate(mention_iter):
            if i % 250000 == 0: 
                log.debug('Processed %i mentions...' % i)
            for url, _, _, _ in mentions:
                occurence_model.add(e, url)

        occurence_model.write(self.out_path)
        log.info('Done')

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('mention_model_path', metavar='MENTION_MODEL')
        p.add_argument('occurrence_model_path', metavar='OCCURRENCE_MODEL')
        p.set_defaults(cls=cls)
        return p

class FilterTermModel(object):
    "Filter model."
    def __init__(self, inpath, outpath):
        self.in_path = inpath
        self.out_path = outpath

    def __call__(self):
        wv = WordVectors.read('/n/schwa11/data0/linking/erd/full/models/googlenews300.wordvector.model')
        vocab = set(wv.vocab.iterkeys())
        wv = None

        log.debug('Loading term model...')

        term_model = mmdict(self.in_path)

        def iter_filted_terms(eds):
            for i, (e, d) in enumerate(eds):
                if i % 100000 == 0:
                    log.debug('Processed %i entities...' % i)

                yield e, {t:c for t,c in d.iteritems() if t in vocab}

        mmdict.write(self.out_path, iter_filted_terms(term_model.iteritems()))

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('inpath', metavar='INFILE')
        p.add_argument('outpath', metavar='OUTFILE')
        p.set_defaults(cls=cls)
        return p

class BuildIdfsForEntityContext(object):
    "Builds term inverse document frequency model from an entity term frequency model."
    def __init__(self, inpath, outpath):
        self.in_path = inpath
        self.out_path = outpath

    def __call__(self):       
        # todo: fix wv_vocab at module scope

        wv = WordVectors.read('/n/schwa11/data0/linking/models/googlenews300.wordvector.model')
        vocab = set(wv.vocab.iterkeys())
        wv = None

        log.debug('Computing dfs over entity context model...')

        dfs = defaultdict(int)
        entity_count = 0
        for i, d in enumerate(mmdict.static_itervalues(self.in_path)):
            if i % 250000 == 0:
                log.debug("Processed %i entities..." % i)
            for t in d.iterkeys():
                if t in vocab:
                    dfs[t] += 1
            entity_count += 1
        
        def iter_term_idfs():
            for t, df in dfs.iteritems():
                yield (t, math.log(entity_count/df))

        log.debug('Writing idf model: %s' % self.out_path)
        mmdict.write(self.out_path, iter_term_idfs())

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('inpath', metavar='INFILE')
        p.add_argument('outpath', metavar='OUTFILE')
        p.set_defaults(cls=cls)
        return p

class ExportEntityInfo(object):
    """ Creates a tsv with entity info useful for autocompletion """
    def __init__(self, entity_set_model_path, entity_model_tag, include_non_entities, out_path):
        self.include_non_entities = include_non_entities
        self.entity_set_model_path = entity_set_model_path
        self.entity_model_tag = entity_model_tag
        self.out_path = out_path

    def is_entity(self, entity_id):
        return not entity_id.lower().endswith('(disambiguation)')

    def __call__(self):
        entity_set = marshal.load(open(self.entity_set_model_path,'r')) 
        prior_model = model.EntityPrior(self.entity_model_tag)
        desc_model = model.EntityDescription()

        total = len(entity_set)
        skipped = 0
        filtered = 0
        log.info('Exporting entity information... (entities only=%s)', str(not self.include_non_entities))
        with open(self.out_path, 'w') as f:
            for i, entity_id in enumerate(entity_set):
                if i % 250000 == 0:
                    log.debug('Processed %i entities...', i)

                info = desc_model.get(entity_id)
                
                if info == None:
                    skipped += 1
                    continue
                if not (self.include_non_entities or self.is_entity(entity_id)):
                    filtered += 1
                    continue

                count = str(prior_model.count(entity_id))
                description = info['description']
                label = info['label']
                row = '\t'.join([label, count, description, entity_id])+'\n'
                f.write(row.encode('utf-8'))
        log.info("Entity export complete, %i total entities, %i skipped, %i filtered", total, skipped, filtered)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('entity_set_model_path', metavar='ENTITY_SET_MODEL_PATH')
        p.add_argument('entity_model_tag', metavar='PRIOR_MODEL_TAG')
        p.add_argument('out_path', metavar='OUT_PATH')
        p.add_argument('--include_non_entities', metavar='INCLUDE_NON_ENTITIES', type=bool, required=False, default=False)
        p.set_defaults(cls=cls)
        return p
