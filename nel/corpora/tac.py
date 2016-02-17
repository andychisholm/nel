#!/usr/bin/env python
import os
import re
from xml.etree import ElementTree
from collections import defaultdict

from .prepare import PrepareCorpus
from ..doc import Doc, Chain, Mention, Candidate
from ..model.corpora import Redirects
from ..process.tag import Tagger, StanfordTagger
from ..process.tokenise import RegexTokeniser, TOKEN_RE
from ..harness.format import markup_to_whitespace

from nel import logging

log = logging.getLogger()
ENC = 'utf8'

@PrepareCorpus.Register
class TacPrepare(object):
    """ Prepare a set of docrep TAC query documents """
    def __init__(self, mentions_path, links_path, docs_path, redirect_model_tag):
        self.mentions_path = mentions_path
        self.links_path = links_path
        self.docs_path = docs_path
        self.redirect_model_tag = redirect_model_tag

    def __call__(self):
        log.info('Joining tac queries and links...')
        mentions_by_id = {m['id']:m for m in self.iter_mentions()}
        for link in self.iter_links():
            mentions_by_id[link['query']]['resolution'] = link['resolution']

        mentions_by_doc = defaultdict(list)
        for m in mentions_by_id.itervalues():
            mentions_by_doc[m['doc']].append(m)

        log.info('Preparing tac documents...')
        for d in self.iter_docs():
            log.info("Preparing doc: %s ...", d['id'])
            doc = Doc(doc_id=d['id'],text=d['text'])
            doc.chains = []

            for m in mentions_by_doc[d['id']]:
                if d['text'][m['span']] != m['text']:
                    if d['text'][m['span']].strip() == '':
                        # in some documents, attributes on html tags have been annotated as entities
                        log.warn('Mention span mismatch - likely annotated markup [%s] (%s)', d['id'], m['text'])
                    else:
                        log.warn("Mention span mismatch [%s] (%s!=%s)", d['id'], d['text'][m['span']], m['text'])

                resolution = Candidate(m['resolution']['id']) if m['resolution']['id'] else None
                doc.chains.append(Chain(
                    mentions=[Mention(m['span'].start, d['text'][m['span']], tag=m['tag'], resolution=resolution)]
                ))

            yield doc

    def iter_mentions(self):
        root = ElementTree.parse(self.mentions_path).getroot()
        for query in root:
            mid = query.attrib['id']
            docid = query.find('docid').text
            txt = query.find('name').text
            begin = int(query.find('beg').text)
            end = int(query.find('end').text) + 1
            yield {
                'id': mid,
                'doc': docid,
                'text': txt,
                'span': slice(begin, end),
                'tag': None
            }

    def iter_links(self):
        redirects = Redirects(self.redirect_model_tag)

        with open(self.links_path, 'r') as f:
            for i, line in enumerate(f):
                if i == 0: continue # skip header

                parts = line.decode(ENC).strip().split('\t')
                cluster = None
                rid = None

                if parts[1].startswith('NIL'):
                    cluster = parts[1][3:]
                else:
                    rid = redirects.map(parts[1])

                yield {
                    'query': parts[0],
                    'resolution': {
                        'id': rid,
                        'cluster': cluster
                    },
                    'type': parts[2]
                }

    def iter_docs(self):
        strip_ctg_nl_re = re.compile(r"([A-Za-z0-9])\n([A-Za-z0-9])")

        for fn in os.listdir(self.docs_path):
            path = os.path.join(self.docs_path, fn)
            with open(path, 'r') as f:
                content = f.read().decode(ENC)
                converted = markup_to_whitespace(content)
                converted = strip_ctg_nl_re.sub(r"\1 \2", converted)
                if len(content) != len(converted):
                    log.error("Markup conversion length mismatch (%i != %i)", len(content), len(converted))

                yield {
                    'id': fn,
                    'text': converted
                }

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('mentions_path', metavar='QUERY_XML_PATH')
        p.add_argument('links_path', metavar='LINKS_TSV_PATH')
        p.add_argument('docs_path', metavar='SOURCE_DOCS_PATH')
        p.add_argument('--redirect_model_tag', default='tac', required=False, metavar='REDIRECT_MODEL')
        p.set_defaults(parsecls=cls)
        return p
