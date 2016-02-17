#!/usr/bin/env python
import sys
import argparse
import textwrap
import json

from pymongo import MongoClient
from flask import Flask, request, Response, abort, make_response
from ..util import parmapper
from ..doc import Doc, Chain, Candidate
from ..harness import format
from ..process import cluster, resolve
from ..process.pipeline import Pipeline

from nel import logging
log = logging.getLogger()

class ServiceHarness(object):
    """ Harness hosting a REST linking endpoint. """
    Instance = None
    App = Flask(__name__)
    def __init__(self, config_path, host, port):
        ServiceHarness.Instance = self
        self.linker = Pipeline.load(config_path)
        self.host = host
        self.port = port

    @staticmethod
    @App.route('/', methods=['POST'])
    def handler():
        return ServiceHarness.Instance.process(request.get_json(True))

    def __call__(self):
        ServiceHarness.App.run(host=self.host, port=self.port)

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
        p.add_argument('config_path', metavar='PIPELINE_CONFIG_PATH')
        p.add_argument('host', default='0.0.0.0', metavar='HOST')
        p.add_argument('port', default=8000, type=int, metavar='PORT')
        return p

class BatchLink(object):
    """ Batch linking harness """
    def __init__(self, corpus, tag, ranker, resolver, clusterer, fmt, output_path):
        self.corpus = corpus
        self.tag = tag
        self.link = None
        self.clusterer = None
        self.output_path = output_path

        if ranker:
            self.link = resolve.FeatureRankResolver(ranker, resolver)
        if self.clusterer:
            self.clusterer = cluster.get(clusterer)

        self.fmt = {
            'neleval':format.to_neleval
        }[fmt]

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('--corpus', metavar='CORPUS', required=True)
        p.add_argument('--tag', metavar='TAG', default=None, required=False)
        p.add_argument('--fmt', metavar='FORMAT', default='neleval', choices=['neleval'], required=False)
        p.add_argument('--ranker', metavar='RANKER', default=None, required=False)
        p.add_argument('--resolver', metavar='RESOLVER', default=None, required=False)
        p.add_argument('--clusterer', metavar='CLUSTERER', default=None, required=False, choices=['name'])
        p.add_argument('--output', dest='output_path', metavar='OUTPUT_PATH', required=True)
        p.set_defaults(cls=cls)
        return p

    def iter_results(self, query):
        for json_doc in query:
            doc = Doc.obj(json_doc)
            if self.link:
                doc = self.link(doc)
            else:
                doc.chains = [Chain(mentions=[m], resolution=m.resolution) for c in doc.chains for m in c.mentions]
            yield doc

    def __call__(self):
        """Link documents """
        store = MongoClient().docs[self.corpus]
        flt = {}
        if self.tag != None:
            flt['tag'] = self.tag
        
        query = store.find(flt)
        log.info(
            'Writing %s linking output for %i docs from %s%s to: %s...',
            'system' if self.link else 'gold',
            query.count(),
            self.corpus, (' '+self.tag) if self.tag else '',
            self.output_path)

        result_iter = self.iter_results(query)
        if self.link and self.clusterer:
            result_iter = self.clusterer(list(self.iter_results(query)))

        with open(self.output_path, 'w') as f:
            for doc in result_iter:
                if doc.chains:
                    f.write(self.fmt(doc).encode('utf-8')+'\n')
