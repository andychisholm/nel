#!/usr/bin/env python
import sys
import argparse
import textwrap
import json

from pymongo import MongoClient
from flask import Flask, request, Response, abort, make_response
from ..util import parmapper
from ..doc import Doc, Candidate
from ..linkers import LINKERS, RankingResolver
from ..harness import format
from ..process import cluster

import logging
log = logging.getLogger()

class ServiceHarness(object):
    """ Harness hosting a REST linking endpoint. """
    Instance = None
    App = Flask(__name__)
    def __init__(self, **kwargs):
        ServiceHarness.Instance = self
        self.host = kwargs.pop('host')
        self.port = int(kwargs.pop('port'))
        self.linker = kwargs.pop('linkcls')(**kwargs)
    
    @staticmethod
    @App.route('/', methods=['POST'])
    def handler():
        return ServiceHarness.Instance.process(request.get_json(True))

    def __call__(self):
        ServiceHarness.App.run(host='0.0.0.0',port=self.port)

    def read(self, doc):
        if doc['type'] == 'text/plain':
            return Doc(text=doc['content'], doc_id=doc['id'])
        elif doc['type'] == 'text/markdown':
            return Doc(text=format.markdown_to_whitespace(doc['content']), doc_id=doc['id'])

    def get_plaintext_document(self, doc):
        return Doc(text=doc['content'], id=doc['id'])

    def process(self, data):
        log.info('Processing document %s...', data['doc']['id'])
        doc = self.linker(self.read(data['doc']))

        if  data['format'] == 'json':
            formatter, mimetype = format.to_json, 'application/json'
        elif data['format'] == 'tsv':
            formatter, mimetype = format.to_neleval, 'text/tab-separated-values'
        else:
            abort(make_response("Invalid or unset response format requested.", 422))

        return Response(formatter(doc).encode('utf-8'),  mimetype=mimetype)
    
    @classmethod
    def add_arguments(cls, p):
        p.add_argument('host', metavar='HOST')
        p.add_argument('port', metavar='PORT')

        sp = p.add_subparsers()
        for linkcls in LINKERS:
            name = linkcls.__name__
            help_str = linkcls.__doc__.split('\n')[0]
            desc = textwrap.dedent(linkcls.__doc__.rstrip())
            csp = sp.add_parser(name,
                                help=help_str,
                                description=desc,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
            linkcls.add_arguments(csp)

        p.set_defaults(cls=cls)
        return p

class BatchLink(object):
    """ Batch linking harness """
    def __init__(self, corpus, tag, ranker, resolver, clusterer, fmt):
        self.corpus = corpus
        self.tag = tag
        self.out_fh = sys.stdout
        
        self.link = None
        self.clusterer = None

        if ranker:
            self.link = RankingResolver(ranker, resolver)
        if self.clusterer:
            self.clusterer = cluster.get(clusterer)

        self.fmt = {
            'neleval':format.to_neleval
        }[fmt]

    def iter_results(self, query):
        for json_doc in query:
            doc = Doc.obj(json_doc)
            if self.link:
                doc = self.clean(doc)
                doc = self.link(doc)
            yield doc

    def __call__(self):
        """Link documents """
        store = MongoClient().docs[self.corpus]
        flt = {}
        if self.tag != None:
            flt['tag'] = self.tag
        
        query = store.find(flt)
        log.info('Linking %i %s%s documents...', query.count(), self.corpus, (' '+self.tag) if self.tag else '')

        result_iter = self.iter_results(query)
        if self.link and self.clusterer:
            result_iter = self.clusterer(list(self.iter_results(query)))

        for doc in result_iter:
            out = self.fmt(doc).encode('utf-8')
            if out:
                print >>self.out_fh, out

        log.info('Done.')

    def clean(self, doc):
        for c in doc.chains:
            c.resolution = None
        return doc

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('--corpus', metavar='CORPUS', required=True)
        p.add_argument('--tag', metavar='TAG', default=None, required=False)
        p.add_argument('--fmt', metavar='FORMAT', default='neleval', choices=['neleval'], required=False)
        p.add_argument('--ranker', metavar='RANKER', default=None, required=False)
        p.add_argument('--resolver', metavar='RESOLVER', default=None, required=False)
        p.add_argument('--clusterer', metavar='CLUSTERER', default=None, required=False, choices=['name'])
        p.set_defaults(cls=cls)
        return p
