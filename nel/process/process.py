#!/usr/bin/env python
from pymongo import MongoClient
from time import time

from ..doc import Doc
from ..util import parmapper

from progressbar import ProgressBar, Bar, Counter, ETA, FileTransferSpeed, Percentage, Timer

import logging
log = logging.getLogger()

class Process(object):
    def __call__(self, doc):
        """Add annotations to doc and return it"""
        raise NotImplementedError

class Pipeline(object):
    def __init__(self, processors):
        self.processors = processors

    def __call__(self, doc):
        for c in self.processors:
            doc = c(doc)
        return doc

class DocMapper(object):
    "Load, process and store documents."
    def __init__(self, **kwargs):
        self.corpus_id = kwargs.pop('corpus')
        self.tag_filter = kwargs.pop('tag')

        self.processes = kwargs.pop('processes')
        if self.processes != None:
            self.processes = int(self.processes)

        self.recycle_interval = kwargs.pop('recycle')
        if self.recycle_interval != None:
            self.recycle_interval = int(self.recycle_interval)

    def mapper(self, doc):
        raise NotImplementedError

    def process_document(self, doc):
        try:
            doc = self.mapper(doc)
        except Exception, e:
            log.warn('Error processing doc (%s): %s', doc.id, str(e))
            raise

        return doc

    def get_corpus_filter(self):
        flt = {}
        if self.tag_filter != None:
            flt['tag'] = self.tag_filter
        return flt

    def count_docs(self, corpus):
        return corpus.count(self.get_corpus_filter())

    def iter_docs(self, corpus):
        cursor = corpus.find(self.get_corpus_filter(), modifiers={'$snapshot':True})
        for json_doc in cursor:
            yield Doc.obj(json_doc)

    def iter_processed_docs(self, corpus):
        try:
            if self.processes == 1:
                for doc in self.iter_docs(corpus):
                    yield self.mapper(doc)
            else:
                with parmapper(self.mapper, nprocs=self.processes,recycle_interval=self.recycle_interval) as pm:
                    for _, doc in pm.consume(self.iter_docs(corpus)):
                        yield doc
        except Exception as e:
            log.warn('Exception during feature extraction: %s', str(e))

    def __call__(self):
        start_time = time()

        corpus = MongoClient().docs[self.corpus_id]
        total_docs = self.count_docs(corpus)

        widgets = [
            'Processed: ', Counter(), '/', str(total_docs), ' ',
            '(', FileTransferSpeed(unit='d'), ') ',
            Bar(marker='#', left='[', right=']'),
            ' ', Percentage(), ' ',
            ETA(),
            ' (', Timer(format='Elapsed: %s'), ')'
        ]

        log.info('Running %s-process doc mapper over %i docs', 'single' if self.processes==1 else 'multi', total_docs)
        with ProgressBar(total_docs, widgets, redirect_stdout=self.processes != 1) as progress:
            for i, doc in enumerate(self.iter_processed_docs(corpus)):
                try:
                    corpus.save(doc.json())
                    progress.update(i)
                except:
                    log.warn('Error saving processed document.')
                    raise
        log.info('Done.')

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('--corpus', metavar='CORPUS')
        p.add_argument('--tag', default=None, required=False, metavar='TAG_FILTER')
        p.add_argument('--processes', default=None, required=False, type=int, metavar='PROCESS_COUNT')
        p.add_argument('--recycle', default=None, required=False, metavar='WORKER_RECYCLE_INTERVAL')
        p.set_defaults(cls=cls)
        return p
