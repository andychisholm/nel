# -*- coding: utf-8 -*-
import argparse
import textwrap
import numpy

from collections import defaultdict
from pymongo import MongoClient

from ..doc import Doc

from nel import logging
log = logging.getLogger()

class CompareCorpusAnnotations(object):
    """ Output summary statistics for documents in a corpus """
    def __init__(self, **kwargs):
        self.gold_corpus_id = kwargs.pop('gold_corpus_id')
        self.gold_corpus_tag = kwargs.pop('gold_corpus_tag')
        self.system_corpus_id = kwargs.pop('system_corpus_id')
        self.system_corpus_tag = kwargs.pop('system_corpus_tag')

    @staticmethod
    def iter_docs(corpus_id, corpus_tag):
        for d in MongoClient()['docs'][corpus_id].find({'tag':corpus_tag}):
            yield Doc.obj(d)

    def format_doc(self, gold, system):
        text = gold.text
        gms = [(m.begin, m.end, 'gold') for c in gold.chains for m in c.mentions]
        sms = [(m.begin, m.end, 'system') for c in system.chains for m in c.mentions]

        raw_spans = sorted(gms + sms)
        spans = []
        for begin, end, tag in raw_spans:
            if spans and begin < spans[-1][1]:
                last_span = spans[-1]
                spans[-1] = (last_span[0], begin, last_span[2])
                spans.append((begin, last_span[1], 'both'))
                begin = last_span[1]
            if begin != end:
                spans.append((begin, end, tag)) 

        output_text = []
        last_offset = None
        for begin, end, tag in spans:
            output_text.append(text[last_offset:begin])
            output_text.append("<span class='" + tag + "'>" + text[begin:end] + "</span>")
            last_offset = end
        text = ''.join(output_text)

        text = text.replace('\n', '</br>')
        return DOC_TEMPLATE % (gold.id, gold.id, text)

    def __call__(self):
        gold_docs = {d.id:d for d in self.iter_docs(self.gold_corpus_id, self.gold_corpus_tag)}
        system_docs = {d.id:d for d in self.iter_docs(self.system_corpus_id, self.system_corpus_tag)}
        output = ROOT_TEMPLATE % u'\n'.join(self.format_doc(gold_docs[docid], system_docs[docid]) for docid in gold_docs.iterkeys())
        print output.encode('utf-8')

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('gold_corpus_id', metavar='GOLD_CORPUS_ID')
        p.add_argument('gold_corpus_tag', metavar='GOLD_CORPUS_TAG')
        p.add_argument('system_corpus_id', metavar='SYSTEM_CORPUS_ID')
        p.add_argument('system_corpus_tag', metavar='SYSTEM_CORPUS_TAG')
        p.set_defaults(cls=cls)
        return p

ROOT_TEMPLATE = u"""
<html>
    <meta charset="UTF-8">
    <head>
        <style>
        div.doc {
            padding: 5px;
            border: 1px solid #000;
            line-height: 130%%;
            margin-bottom: -1px;
        }
        div.doc h3 {
          background-color: #ddd;
          padding: 0px 4px;
          margin: 0;
        }
        span {
            padding: 0px 0px;
        }
        span.gold {
            background-color: #cceeff;
        }
        span.system {
            background-color: #ffbbbb;
        }
        span.both {
            background-color: #bbeebb;
        }
        </style>
    </head>
    <body>
        %s
    </body>
</html>
"""
DOC_TEMPLATE = u"""
        <div class='doc'>
            <h3 id="%s">%s</h3>
            %s
        </div>
"""
