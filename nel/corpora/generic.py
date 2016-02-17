#!/usr/bin/env python
import os
from collections import defaultdict

from .prepare import PrepareCorpus
from ..doc import Doc, Chain, Mention, Candidate
from ..model.corpora import Redirects
from ..process.tokenise import RegexTokeniser, TOKEN_RE
from ..harness.format import markdown_to_whitespace

from nel import logging
log = logging.getLogger()

ENC = 'utf8'
def normalise_wikipedia_link(s):
    s = s.replace(' ', '_').strip('_').strip()
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return trim_link_subsection(s)

def trim_link_subsection(s):
    idx = s.find('#')
    return s if idx == -1 else s[:idx]

@PrepareCorpus.Register
class MarkdownPrepare(object):
    """ Injest a set of markdown documents with neleval formatted annotations """
    def __init__(self, docs_path, annotations_path, redirect_model_tag, target_entity_filter):
        self.docs_path = docs_path
        self.annotations_path = annotations_path
        self.redirect_model = Redirects(redirect_model_tag)
        self.target_entity_filter = target_entity_filter

    def iter_mentions(self):
        with open(self.annotations_path, 'r') as f:
            for line in f:
                parts = line.decode(ENC).strip().split('\t')

                tag = None
                resolution_id = None
                if len(parts) > 3 and parts[3].strip():
                    resolution_id = 'en.wikipedia.org/wiki/' + normalise_wikipedia_link(parts[3])
                    resolution_id = self.redirect_model.map(resolution_id)
                if len(parts) > 5:
                    tag = parts[5].lower().strip()

                if self.target_entity_filter and resolution_id:
                    if not resolution_id.startswith(self.target_entity_filter):
                        resolution_id = None

                yield {
                    'doc': parts[0],
                    'span': slice(int(parts[1]), int(parts[2])),
                    'tag': tag,
                    'resolution': {
                        'id': resolution_id
                    }
                }

    def iter_docs(self, docids):
        for fn in os.listdir(self.docs_path):
            if fn.startswith('.'):
                continue
            path = os.path.join(self.docs_path, fn)
            docid = fn.split('.')[0]
            if docid not in docids:
                continue
            with open(path, 'r') as f:
                content = f.read().decode(ENC)
                converted = markdown_to_whitespace(content)
                if len(content) != len(converted): # or docid == '04cc61da-0cbb-44f1-94c9-5d3daff887b2':
                    log.error('Markdown to whitespace offset mismatch.')
                    import code
                    code.interact(local=locals())
                yield {
                    'id': docid,
                    'text': converted,
                    'raw': content
                }

    def __call__(self):
        log.info('Processing annotation set...')
        mentions_by_doc = defaultdict(list)
        for m in self.iter_mentions():
            mentions_by_doc[m['doc']].append(m)

        for d in self.iter_docs(set(mentions_by_doc.iterkeys())):
            doc = Doc(doc_id=d['id'], text=d['text'], raw=d['raw'])

            # todo: need custom logic here depending on the corpus
            doc.tag = 'dev' if doc.id[-1] in '0123' else 'train'

            mentions = []
            for m in mentions_by_doc[d['id']]:
                resolution = m['resolution']['id']
                mentions.append(Mention(
                    begin=m['span'].start,
                    text=d['text'][m['span']],
                    resolution=Candidate(resolution) if resolution else None,
                    tag=m['tag']))
            doc.chains = [Chain(mentions=[m]) for m in mentions]
            yield doc

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('docs_path', metavar='SOURCE_DOCS_PATH')
        p.add_argument('annotations_path', metavar='ANNOTATIONS_TSV_PATH')
        p.add_argument('--redirect-model-tag', dest='redirect_model_tag', default='wikipedia', required=False, metavar='REDIRECT_MODEL')
        p.add_argument('--target-entity-filter', dest='target_entity_filter', default=None, required=False, metavar='FILTER')
        p.set_defaults(parsecls=cls)
        return p
