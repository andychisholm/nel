import re

from collections import defaultdict
from .prepare import PrepareCorpus
from ..doc import Doc, Candidate, Chain, Mention
from ..model.corpora import Redirects
from ..process import tag
from ..process.tokenise import RegexTokeniser

from nel import logging
log = logging.getLogger()

DOCSTART_MARKER = '-DOCSTART-'

@PrepareCorpus.Register
class ConllPrepare(object):
    """ Tokenise and tag a conll document """
    def __init__(self, conll_data_path, doc_type, redirect_model_tag):
        self.in_path = conll_data_path
        self.doc_predicate = {
            'all': lambda _: True,
            'train': self.is_training_doc,
            'dev': self.is_dev_doc,
            'test': self.is_test_doc
        }[doc_type]
        self.redirect_model_tag = redirect_model_tag

    @staticmethod
    def is_training_doc(doc_id):
        return 'test' not in doc_id
    @staticmethod
    def is_test_doc(doc_id):
        return 'testb' in doc_id
    @staticmethod
    def is_dev_doc(doc_id):
        return 'testa' in doc_id

    def __call__(self):
        """ Prepare documents """
        redirects = Redirects(self.redirect_model_tag)

        log.info('Preparing conll documents...')
        for doc, conll_tags in self.iter_docs(self.in_path, self.doc_predicate):
            doc.chains = []
            for gold_entity_id, (start, end) in conll_tags:
                resolution = Candidate(redirects.map('en.wikipedia.org/wiki/' + gold_entity_id))
                doc.chains.append(Chain(mentions=[
                    Mention(start, doc.text[start:end], resolution=resolution)
                ]))

            yield doc

    @staticmethod
    def iter_docs(path, doc_id_predicate, redirect_model = None, max_docs = None):
        redirect_model = redirect_model or {}

        with open(path, 'rd') as f:
            doc_id = None
            doc_tokens = None
            doc_mentions = None
            doc_count = 0
            for line in f:
                parts = line.decode('utf-8').split('\t')
                if len(parts) > 0:
                    token = parts[0].strip()

                    # if this line contains a mention
                    if len(parts) >= 4 and parts[1] == 'B':
                        # filter empty and non-links
                        if parts[3].strip() != '' and not parts[3].startswith('--'):
                            entity = parts[3]
                            entity = redirect_model.get(entity, entity)
                            begin = sum(len(t)+1 for t in doc_tokens)

                            dodgy_tokenisation_bs_offset = 1 if re.search('[A-Za-z],',parts[2]) else 0

                            position = (begin, begin + len(parts[2]) + dodgy_tokenisation_bs_offset)
                            doc_mentions.append((entity, position))

                    if token.startswith(DOCSTART_MARKER):
                        if doc_id != None and doc_id_predicate(doc_id):
                            doc_count += 1
                            yield Doc(' '.join(doc_tokens), doc_id, ConllPrepare.doc_tag_for_id(doc_id)), doc_mentions

                            if max_docs != None and doc_count >= max_docs:
                                doc_id = None
                                break 
                                
                        doc_id = token[len(DOCSTART_MARKER)+2:-1]
                        doc_tokens = []
                        doc_mentions = []
                    elif doc_id != None:
                        doc_tokens.append(token)

            if doc_id != None and doc_id_predicate(doc_id):
                yield Doc(' '.join(doc_tokens), doc_id, ConllPrepare.doc_tag_for_id(doc_id)), doc_mentions

    @staticmethod
    def doc_tag_for_id(doc_id):
        if 'testa' in doc_id:
            return 'dev'
        elif 'testb' in doc_id:
            return 'test' 
        return 'train'

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('conll_data_path', metavar='CONLL_DATA')
        p.add_argument('doc_type', metavar='DOC_TYPE', choices='all,train,dev,test')
        p.add_argument('--redirect_model_tag', required=False, default='wikipedia', metavar='REDIRECT_MODEL_TAG')
        p.set_defaults(parsecls=cls)
        return p
