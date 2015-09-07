#!/usr/bin/env python
import math

import operator
import unicodedata
import ujson as json

from pymongo import MongoClient
from collections import defaultdict, Counter

from .. import model

import logging
log = logging.getLogger()

class ExportEntityInfo(object):
    """ Exports a tsv file with entity info useful in autocompletion """
    def __init__(self, entities_model_tag, entity_prior_model_tag, threshold, out_path):
        self.entities_model_tag = entities_model_tag
        self.entity_prior_model_tag = entity_prior_model_tag
        self.entity_prior_threshold = threshold
        self.out_path = out_path

    def normalise_label(self, label):
        return unicodedata.normalize('NFKD', label).encode('ascii','ignore')

    def __call__(self):
        entities_model = model.Entities(self.entities_model_tag)
        prior_model = model.EntityPrior(self.entity_prior_model_tag)

        total = 0
        filtered = 0
        missing_description = 0
        log.info('Exporting entity information...')
        with open(self.out_path, 'w') as f:
            for i, (eid, label, description, _) in enumerate(entities_model.iter_entities()):
                if i % 250000 == 0:
                    log.debug('Processed %i entities...', i)

                count = prior_model.count(eid)
                if count < self.entity_prior_threshold:
                    filtered += 1
                    continue

                total += 1
                if not description:
                    missing_description += 1

                label = self.normalise_label(label)
                row = '\t'.join([label, str(count), description, eid])+'\n'
                f.write(row.encode('utf-8'))

        log.info("Completed export of %i entities (%i filtered, %i missing description)", total, filtered, missing_description)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('entities_model_tag', metavar='ENTITIES_MODEL_TAG')
        p.add_argument('entity_prior_model_tag', metavar='PRIOR_MODEL_TAG')
        p.add_argument('out_path', metavar='OUT_PATH')
        p.add_argument('--threshold', metavar='ENTITY_COUNT_THRESHOLD', type=int, default=1, required=False)
        p.set_defaults(cls=cls)
        return p

class ExportContextTrainingSet(object):
    """ Building a training set suitable for training ranking problem over context features """
    def __init__(self, corpus, tag, ctx_path, out_path):
        NUM_TERMS = 100000
        self.store = MongoClient()['docs'][corpus]
        self.tag_filter = tag
        self.out_path = out_path

        self.dfs = model.mmdict(ctx_path + '.df.model')
        log.info('Calculating top %i df terms for use as features...', NUM_TERMS)
        self.dfs = sorted(self.dfs.iteritems(),key=operator.itemgetter(1),reverse=True)[:NUM_TERMS]

        self.feature_terms = {t:i for i,(t,_) in enumerate(self.dfs)}
        self.dfs = dict(self.dfs)
        self.entity_tfs = model.mmdict(ctx_path + '.tf.model')

    def idf(self, term):
        N = 4638072.0
        df = self.dfs[term]
        return math.log(N/df)

    def iter_docs(self):
        from ...doc import Doc
        flt = {}
        if self.tag_filter != None:
            flt['tag'] = self.tag_filter
        
        cursor = self.store.find(flt, snapshot=True)

        log.debug("Found %i documents...", cursor.count())
        for json_doc in cursor:
            yield Doc.obj(json_doc)

    def entity_fvr(self, entity_id):
        return self.bow_to_fvr(self.entity_tfs[entity_id] or {})

    def bow_to_fvr(self, bow):
        fvr = []
        for t,count in bow.iteritems():
            idx = self.feature_terms.get(t, None)
            if idx != None:
                idf = self.idf(t)
                fvr.append([idx,math.sqrt(count)*idf])
        return fvr

    def __call__(self):
        log.info('Writing instances to file: %s ...', self.out_path)
        with open(self.out_path, 'w') as f:
            for i, doc in enumerate(self.iter_docs()):
                if i % 50 == 0:
                    log.info("Processed %i docs...", i)

                p2ns = defaultdict(set)
                positives = []
                negatives = []

                for mention in doc.chains:
                    if mention.resolution == None or mention.resolution.id == None:
                        continue
                    ns = set(c.id for c in mention.candidates if c.id != mention.resolution.id)
                    p2ns[mention.resolution.id] = p2ns[mention.resolution.id].union(ns)

                for p, ns in p2ns.iteritems():
                    positives.append(self.entity_fvr(p))

                    ns_tfs = [self.entity_fvr(n) for n in ns]
                    ns_tfs = sorted(ns_tfs, key=len,reverse=True)
                    negatives.append(ns_tfs)

                f.write(json.dumps({
                    'doc': self.bow_to_fvr(Counter(t.lower() for t in doc.text.split())),
                    'positives': positives,
                    'negatives': negatives
                }) + '\n')

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('--corpus', metavar='CORPUS')
        p.add_argument('--tag', default=None, required=False, metavar='TAG_FILTER')
        p.add_argument('ctx_path', metavar='CTX_MODEL')
        p.add_argument('out_path', metavar='OUT_PATH')
        p.set_defaults(cls=cls)
        return p
