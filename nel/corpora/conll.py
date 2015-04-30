import re

from collections import defaultdict
from .prepare import PrepareCorpus
from ..doc import Doc, Candidate, Chain, Mention
from ..model import model
from ..process import tag
from ..process.tokenise import RegexTokeniser

import logging
log = logging.getLogger()

DOCSTART_MARKER = '-DOCSTART-'

@PrepareCorpus.Register
class ConllPrepare(object):
    """ Tokenise and tag a conll document """
    def __init__(self, conll_data_path, doc_type):
        self.in_path = conll_data_path
        self.doc_predicate = {
            'all': lambda _: True,
            'train': self.is_training_doc,
            'dev': self.is_dev_doc,
            'test': self.is_test_doc
        }[doc_type]

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
        # todo: parameterise model tags
        redirects = model.Redirects('wikipedia')
        candidates = model.Candidates('wikipedia')

        total_mentions = 0
        total_recalled_mentions = 0
        
        log.info('Preparing conll documents...')
        for doc, conll_tags in self.iter_docs(self.in_path, self.doc_predicate):
            mentions = []
            for gold_entity, (start, end) in conll_tags:
                m = Mention(start, doc.text[start:end])
                m.resolution = redirects.map(gold_entity)
                m.surface_form = doc.text[start:end].replace(' ,', ',')
                mentions.append(m)

            chains = tag.Tagger.cluster_mentions(mentions)
            unique_chains = []

            # join chains with the same gold standard resolution, split chains with different gold resolution
            # this requires gold annot information so is only valid if building the train set or isolating disambiguation performance
            for chain in chains:
                mbr = defaultdict(list)
                for m in chain.mentions:
                    mbr[m.resolution].append(m)
                for r, ms in mbr.iteritems():
                    unique_chains.append(Chain(mentions=ms, resolution=Candidate(r)))

            doc.chains = unique_chains
 
            for chain in doc.chains:
                total_mentions += len(chain.mentions)
                # longest mention string
                sf = sorted(chain.mentions, key=lambda m: len(m.surface_form), reverse=True)[0].surface_form

                cs = candidates.search(sf)
                if cs:
                    if chain.resolution.id not in cs:
                        log.warn('Entity (%s) not in candidate set for (%s) in doc (%s).', chain.resolution.id, sf, doc.id)
                    else:
                        total_recalled_mentions += len(chain.mentions)
                    chain.candidates = [Candidate(e) for e in cs]
                else:
                    log.warn('Missing alias (%s): %s' % (doc.id, sf))
            yield doc
        log.info("Candidate Recall = %.1f%", float(total_recalled_mentions*100)/total_mentions)

    @staticmethod
    def iter_docs(path, doc_id_predicate, redirect_model = None, max_docs = None):
        """ Read AIDA-CoNLL formatted documents """
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
        p.set_defaults(parsecls=cls)
        return p
