#!/usr/bin/env python
import argparse
import textwrap
import cPickle as pickle
import numpy

from pymongo import MongoClient
from time import time

from .feature import FEATURE_SET
from ..doc import Doc
from ..util import parmapper

import logging
log = logging.getLogger()

class ExtractFeature(object):
    "Extract features from a prepared document model."
    FEATURES = FEATURE_SET

    def __init__(self, **kwargs):
        self.corpus_id = kwargs.pop('corpus')
        self.tag_filter = kwargs.pop('tag')
        
        self.processes = kwargs.pop('processes')
        if self.processes != None:
            self.processes = int(self.processes)
        
        self.recycle_interval = kwargs.pop('recycle')
        if self.recycle_interval != None:
            self.recycle_interval = int(self.recycle_interval)
        
        self.extractor = kwargs.pop('featurecls')(**kwargs)

    def extract_document_features(self, doc):
        try:
            doc = self.extractor(doc)
        except Exception, e:
            log.warn('Feature extractor exception: %s', str(e))

        return doc

    def iter_docs(self, corpus):
        flt = {}
        if self.tag_filter != None:
            flt['tag'] = self.tag_filter
        
        cursor = corpus.find(flt, snapshot=True)

        log.debug("Found %i documents for feature extraction...", cursor.count())
        for json_doc in cursor:
            yield Doc.obj(json_doc)

    def iter_processed_docs(self, corpus):
        try:
            if self.processes == 1:
                # debugging
                log.info("Starting single-process feature extraction...")
                for doc in self.iter_docs(corpus):
                    yield self.extract_document_features(doc)
            else:
                with parmapper(self.extract_document_features, nprocs=self.processes,recycle_interval=self.recycle_interval) as pm:
                    log.info("Starting parallel feature extraction (%i procs)...", len(pm.procs))
                    for _, doc in pm.consume(self.iter_docs(corpus)):
                        yield doc
        except Exception, e:
            log.warn('Exception during feature extraction: %s', str(e))
    
    def __call__(self): 
        # track performance statistics of the feature extraction process
        total_docs = 0
        total_chains = 0
        total_candidates = 0
        start_time = time()
    
        corpus = MongoClient().docs[self.corpus_id]
        for doc in self.iter_processed_docs(corpus):
            total_docs += 1
            total_chains += len(doc.chains)
            total_candidates += sum(len(m.candidates) for m in doc.chains)

            if total_docs % 100 == 0:
                duration = float(time() - start_time)
                log.info(
                    'Extracted feature for %i docs... ( %.2f d/s %.2f ch/s %.2f c/s )',
                    total_docs,
                    total_docs/duration,
                    total_chains/duration,
                    total_candidates/duration)
            try:
                corpus.save(doc.json())
            except:
                log.warn('Error saving processed document')
        
        log.info('Done.')
        
    @classmethod
    def add_arguments(cls, p):
        p.add_argument('--corpus', metavar='CORPUS')
        p.add_argument('--tag', default=None, required=False, metavar='TAG_FILTER')
        p.add_argument('--processes', default=None, required=False, metavar='PROCESS_COUNT')
        p.add_argument('--recycle', default=None, required=False, metavar='WORKER_RECYCLE_INTERVAL')
        p.set_defaults(cls=cls)

        sp = p.add_subparsers()
        for featurecls in cls.FEATURES:
            name = featurecls.__name__
            help_str = featurecls.__doc__.split('\n')[0]
            desc = textwrap.dedent(featurecls.__doc__.rstrip())
            csp = sp.add_parser(name,
                                help=help_str,
                                description=desc,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
            featurecls.add_arguments(csp)

        return p
