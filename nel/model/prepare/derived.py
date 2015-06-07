#!/usr/bin/env python
import math
import re
import logging
import marshal
import os
import operator
import unicodedata

from time import time
from collections import defaultdict, Counter
from schwa import dr
from .. import model
from .. import data
from ..model import WordVectors
from ..model import EntityOccurrence, EntityCooccurrence
from ..model import Candidates

from .util import trim_subsection_link, normalise_wikipedia_link
from ...util import parmapper

log = logging.getLogger()

class MRCorpusProcessor(object):
    def __init__(self, docs_path, doc_schema):
        self.docs_path = docs_path
        self.doc_schema = doc_schema

    def process_chunk(self, path):
        state = []
        with open(path,'r')  as f:
            log.debug("Processing chunk: %s ...", path)
            for doc in dr.Reader(f, self.doc_schema):
                state.append(self.mapper(doc))
        return self.combiner(state)

    def mapper(self, doc):
        raise NotImplementedError

    def combiner(self, results):
        return results

    def iter_file_names(self):
        for path, _, files in os.walk(self.docs_path):
            for filename in sorted(files):
                if filename.endswith('.dr'):
                    yield os.path.join(path, filename)

    def iter_results(self, processes=None):
        if processes == 1:
            for path in self.iter_file_names():
                for result in self.process_chunk(path):
                    yield result
        else:
            with parmapper(self.process_chunk, nprocs=processes, recycle_interval=None) as pm:
                for _, results in pm.consume(self.iter_file_names()):
                    for result in results:
                        yield result

class BuildLinkModels(MRCorpusProcessor):
    "Build link derived models from a docrep corpus."
    def __init__(self, docs_path, redirect_model_tag, entities_model_tag, model_tag):
        class Link(dr.Ann):
            anchor = dr.Field()
            target = dr.Field()
        class Doc(dr.Doc):
            name = dr.Field()
            links = dr.Store(Link)

        super(BuildLinkModels, self).__init__(docs_path, Doc.schema())

        self.model_tag = model_tag
        self.entities_model_tag = entities_model_tag
        self.redirects = model.Redirects(redirect_model_tag, prefetch=False) #True)

        entity_model = model.Entities(self.entities_model_tag)

        log.info('Loading entity set...')
        self.entity_set = set(entity_model.iter_ids())

        if self.entity_set:
            log.info("Building link models over %i entities...", len(self.entity_set))
        else:
            raise Exception("Entity set (%s) is empty, build will not yield results.", self.entities_model_tag)

    def mapper(self, doc):
        name_entity_pairs = []

        for link in doc.links:
            # we may want to ignore subsection links when generating name models
            # sometimes a page has sections which refer to subsidiary entities
            # links to these may dilute the name posterior for the main entity
            # for now we just add everything to keep it simple
            target = trim_subsection_link(link.target)
            target = normalise_wikipedia_link(target)
            target = self.redirects.map(target)
            target = trim_subsection_link(target)

            # ignore out-of-kb links in entity and name probability models
            if target in self.entity_set:
                name_entity_pairs.append((link.anchor.lower(), target))

        return name_entity_pairs

    def __call__(self):
        log.info('Building page link derived models from: %s', self.docs_path)

        nep_model = model.NameProbability(self.model_tag)
        log.info("Flushing name model counts...")
        nep_model.store.flush()

        #co_model = EntityCooccurrence(self.model_tag, store_uri='mongodb://localhost')
        #log.info("Flushing cooccurrence counts...")
        #co_model.store.flush()

        entity_counts = Counter()
        with data.BatchedOperation(nep_model.merge, 1000000) as nep_merger:
            for i, name_entity_pairs in enumerate(self.iter_results()):
                if i == 10000 or i % 250000 == 0:
                    log.info('Processed %i documents...', i)

                entity_counts.update(e for _, e in name_entity_pairs)
                for pair in name_entity_pairs:
                    nep_merger.append(pair)

        ep_model = model.EntityPrior(self.model_tag)
        ep_model.create(entity_counts.iteritems())
        log.info('Done')

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('docs_path', metavar='DOCS_PATH')
        p.add_argument('redirect_model_tag', metavar='REDIRECT_MODEL_TAG')
        p.add_argument('entities_model_tag', metavar='ENTITIES_MODEL_TAG')
        p.add_argument('model_tag', metavar='MODEL_TAG')
        p.set_defaults(cls=cls)
        return p

class BuildTermDocumentFrequencyModel(MRCorpusProcessor):
    "Build link derived models from a docrep corpus."
    def __init__(self, docs_path, model_tag):
        self.model_tag = model_tag

        class Token(dr.Ann):
            norm = dr.Field()
        class Doc(dr.Doc):
            name = dr.Field()
            tokens = dr.Store(Token)

        super(BuildTermDocumentFrequencyModel, self).__init__(docs_path, Doc.schema())

    def mapper(self, doc):
        return set(t.norm.lower() for t in doc.tokens)

    def __call__(self):
        log.info('Building doc frequency model from: %s', self.docs_path)
        start_time = time()

        dfs = Counter()
        total_docs = 0
        for i, terms in enumerate(self.iter_results()):
            dfs.update(terms)
            total_docs += 1
            if i % 250000 == 0:
                log.info('Processed %i documents...', i)

        log.info('Storing %i term dfs...', len(dfs))
        df_model = model.TermDocumentFrequency(self.model_tag)
        df_model.create(total_docs, dfs.iteritems())
        log.info('Done')

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('docs_path', metavar='DOCS_PATH')
        p.add_argument('model_tag', metavar='MODEL_TAG')
        p.set_defaults(cls=cls)
        return p

class BuildTermFrequencyModel(MRCorpusProcessor):
    "Build link derived models from a docrep corpus."
    def __init__(self, docs_path, model_tag, vocab_limit = 100000):
        self.model_tag = model_tag

        class Token(dr.Ann):
            norm = dr.Field()
        class Doc(dr.Doc):
            name = dr.Field()
            tokens = dr.Store(Token)

        super(BuildTermFrequencyModel, self).__init__(docs_path, Doc.schema())

        log.info('Getting top %i terms by document frequency...', vocab_limit)
        df_model = model.TermDocumentFrequency(self.model_tag)
        self.terms = set(df_model.iter_most_frequent_terms(vocab_limit))
        log.info('Completed fetch of vocab with %i terms...', len(self.terms))

    def mapper(self, doc):
        return doc.name, Counter(t.norm.lower() for t in doc.tokens if t.norm.lower() in self.terms)

    def __call__(self):
        log.info('Building ctx models from: %s', self.docs_path)
        start_time = time()

        tf_model = model.EntityTermFrequency(self.model_tag, uri='mongodb://localhost')
        log.info('Flushing existing tf counts...')
        tf_model.store.flush()

        dfs = Counter()
        with data.BatchedOperation(tf_model.merge, 25000) as tf_merger:
            for i, (name, bow) in enumerate(self.iter_results()):
                tf_merger.append((name,bow))
                if i % 250000 == 0:
                    log.info('Processed %i documents...', i)
        log.info('Done')

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('docs_path', metavar='DOCS_PATH')
        p.add_argument('model_tag', metavar='MODEL_TAG')
        p.add_argument('--vocab_limit', required=False, default=100000, type=int, metavar='VOCAB_LIMIT')
        p.set_defaults(cls=cls)
        return p

class BuildCandidateModel(object):
    "Builds a model mapping aliases to entites for candidate generation"
    def __init__(self, entities_model_tag, redirect_model_tag, name_model_tag, model_tag):
        self.name_model_tag = name_model_tag
        self.model_tag = model_tag
        self.underscores_re = re.compile('_+')

        self.redirects = model.Redirects(redirect_model_tag, prefetch=True)

        log.info('Pre-fetching kb entity set...')
        self.entities_model = model.Entities(entities_model_tag)
        self.entity_set = set(self.redirects.map(e) for e in self.entities_model.iter_ids())
 
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
        # Include some predefined alias set, e.g. yago-means
        #log.info('Loading aliases: %s ...', self.alias_model_path)
        #alias_model = marshal.load(open(self.alias_model_path,'rb'))
        #log.info('Enumerating aliases for %i entities...' % len(alias_model))
        #for entity, names in alias_model.iteritems():
        #    entity = redirects.get(entity, entity)
        #    for name in names:
        #        yield entity, name
        #alias_model = None

        log.info('Enumerating mappings from titles and aliases in entities model...')
        for eid, label, _, aliases in self.entities_model.iter_entities():
            eid = self.redirects.map(eid)
            if self.include_entity(eid):
                names = set([label, self.convert_title_to_name(eid)] + aliases)
                for name in list(names):
                    yield eid, name
                    normed = unicodedata.normalize('NFKD', name).encode('ascii','ignore')
                    if normed not in names:
                        yield eid, normed
                        names.add(normed)

        log.info('Enumerating redirect titles...')
        for source, target in self.redirects.cache.iteritems():
            if self.include_entity(target):
                yield target, self.convert_title_to_name(source)

        log.info('Enumerating mappings in name probability model...')
        name_model = model.NameProbability(self.name_model_tag)
        for name, entities_iter in name_model.iter_name_entities():
            for eid in entities_iter:
                eid = self.redirects.map(eid)
                if self.include_entity(eid):
                    yield eid, name

    def include_entity(self, entity):
        return entity in self.entity_set

    def __call__(self):
        log.info('Building candidate model...')
        Candidates(self.model_tag).create(self.iter_entity_aliases())
        log.info("Done.")

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('entities_model_tag', metavar='ENTITIES_MODEL_TAG')
        p.add_argument('redirect_model_tag', metavar='REDIRECT_MODEL_TAG')
        p.add_argument('name_model_tag', metavar='NAME_MODEL_TAG')
        p.add_argument('model_tag', metavar='CANDIDATE_MODEL_TAG')
        p.set_defaults(cls=cls)
        return p

from bisect import bisect_right
class BuildMentionModel(MRCorpusProcessor):
    "Build mention model over a link document corpus"
    def __init__(self, docs_path):
        class Token(dr.Ann):
            norm = dr.Field()
        class Link(dr.Ann):
            span = dr.Slice(Token)
            target = dr.Field()
        class Sentence(dr.Ann):
            span = dr.Slice(Token)
        class Doc(dr.Doc):
            name = dr.Field()
            tokens = dr.Store(Token)
            links = dr.Store(Link)
            sentences = dr.Store(Sentence)

        super(BuildMentionModel, self).__init__(docs_path, Doc.schema())
        self.redirect_model = model.Redirects('wikipedia', prefetch=False)

    def normalise_target(self, target):
        return self.redirect_model.map(normalise_wikipedia_link(target))

    def mapper(self, doc):
        sentence_offsets = [s.span.start for s in doc.sentences]
        links_by_sentence = defaultdict(list)
        for link in doc.links:
            sentence_idx = bisect_right(sentence_offsets, link.span.start) - 1
            links_by_sentence[sentence_idx].append(link)

        mentions = []
        for sidx, links in links_by_sentence.iteritems():
            sentence = [t.norm for t in doc.tokens[doc.sentences[sidx].span]]
            links = [[link.span, self.normalise_target(link.target)] for link in links]
            mentions.append((sentence, links))

        for toks, links in mentions:
            print ' '.join(toks)
            for span, target in links:
                print '\t' + target

        import code
        code.interact(local=locals())
        return mentions

    def __call__(self):
        log.info("Processing docs: %s ...", self.docs_path)
        for i, name in enumerate(self.iter_results()):
            if i % 10000 == 0:
                log.info(i)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('docs_path', metavar='DOCS_PATH')
        p.set_defaults(cls=cls)
        return p
