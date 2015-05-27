#!/usr/bin/env python
import os
from collections import defaultdict

from .prepare import PrepareCorpus
from ..doc import Doc, Chain, Mention, Candidate
from ..model.model import Redirects
from ..process.tag import Tagger, StanfordTagger, CandidateGenerator
from ..process.tokenise import RegexTokeniser, TOKEN_RE
from ..harness.format import markdown_to_whitespace
from ..model.prepare.util import normalise_wikipedia_link

import logging
log = logging.getLogger()

ENC = 'utf8'

@PrepareCorpus.Register
class MarkdownPrepare(object):
    """ Injest a set of markdown documents with neleval formatted annotations """
    def __init__(self, docs_path, annotations_path, redirect_model_tag, use_gold_mentions):
        self.docs_path = docs_path
        self.annotations_path = annotations_path
        self.redirect_model = Redirects(redirect_model_tag)
        self.use_gold_mentions = use_gold_mentions

    def iter_mentions(self):
        with open(self.annotations_path, 'r') as f:
            for line in f:
                parts = line.decode(ENC).strip().split('\t')

                resolution_id = None
                if len(parts) > 3:
                    resolution_id = normalise_wikipedia_link(parts[3])
                    resolution_id = self.redirect_model.map(resolution_id)

                yield {
                    'doc': parts[0],
                    'span': slice(int(parts[1]), int(parts[2])),
                    'resolution': {
                        'id': resolution_id
                    }
                }

    def iter_docs(self):
        for fn in os.listdir(self.docs_path):
            if fn.startswith('.'):
                continue
            path = os.path.join(self.docs_path, fn)
            with open(path, 'r') as f:
                content = f.read().decode(ENC)
                converted = markdown_to_whitespace(content)
                yield {
                    'id': fn.split('.')[0],
                    'text': converted
                }

    def __call__(self):
        log.info('Processing annotation set...')
        mentions_by_doc = defaultdict(list)
        for m in self.iter_mentions():
            mentions_by_doc[m['doc']].append(m)

        # todo: parameterise tagger/candidate gen config
        generate_candidates = CandidateGenerator('wikipedia')
        if not self.use_gold_mentions:
            tokeniser = RegexTokeniser(TOKEN_RE)
            tagger = StanfordTagger(host='127.0.0.1', port=1447)

        for d in self.iter_docs():
            log.info("Preparing doc: %s ...", d['id'])
            doc = Doc(doc_id=d['id'],text=d['text'])
            mentions = []
            for m in mentions_by_doc[d['id']]:
                mention = Mention(m['span'].start, d['text'][m['span']])
                mention.resolution = m['resolution']['id']
                mentions.append(mention)

            if self.use_gold_mentions:
                doc.chains = Tagger.cluster_mentions(mentions)
                unique_chains = []
                for chain in doc.chains:
                    mbr = defaultdict(list)
                    for m in chain.mentions:
                        mbr[m.resolution].append(m)
                    for r, ms in mbr.iteritems():
                        rc = Candidate(r) if r else None
                        unique_chains.append(Chain(mentions=ms, resolution=rc))
                doc.chains = unique_chains
            else:
                doc = tagger(tokeniser(doc))

            doc = generate_candidates(doc)
            doc.tag = 'dev' # todo: need custom logic here

            yield doc

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('docs_path', metavar='SOURCE_DOCS_PATH')
        p.add_argument('annotations_path', metavar='ANNOTATIONS_TSV_PATH')
        p.add_argument('--redirect_model_tag', default='wikipedia', required=False, metavar='REDIRECT_MODEL')
        p.add_argument('--use_gold_mentions', action='store_true')
        p.set_defaults(use_gold_mentions=False)
        p.set_defaults(parsecls=cls)
        return p
