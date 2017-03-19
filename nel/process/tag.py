#!/usr/bin/env python
import os
import sys
import re
import string

from itertools import izip
from time import time
from bisect import bisect_left
from subprocess import Popen, PIPE
from cStringIO import StringIO

import spacy

from .process import Process
from ..model import recognition
from ..doc import Mention, Chain, Candidate
from ..util import group, spanset_insert, tcp_socket, byte_to_char_map

from nel import logging
log = logging.getLogger()

class Tagger(Process):
    """ Tags and performs naive coref over mentions in tokenised text. """
    def __call__(self, doc):
        doc.chains = [Chain(mentions=[m]) for m in self.tag(doc)]
        return doc

    def tag(self, doc):
        raise NotImplementedError

    def mention_over_tokens(self, doc, i, j, tag=None):
        toks = doc.tokens[i:j]
        begin = toks[0].begin
        end = toks[-1].end
        text = doc.text[begin:end]
        return Mention(begin, text, tag)

    @classmethod
    def iter_options(cls):
        for c in globals().itervalues():
            if c != cls and isinstance(c, type) and issubclass(c, cls):
                yield c

class SpacyTagger(Tagger):
    def __init__(self, spacy_model = None):
        self.spacy_model = spacy_model or 'en_default'
        log.debug('Using spacy entity tagger (%s)...', spacy_model)
        self.nlp = spacy.load(self.spacy_model)

    def tag(self, doc):
        spacy_doc = self.nlp(doc.text)
        doc.tokens = [Mention(t.idx, t.text) for t in spacy_doc]

        for ent in spacy_doc.ents:
            tok_idxs = [i for i in xrange(len(ent)) if not ent[i].is_space]
            if tok_idxs:
                yield self.mention_over_tokens(doc, ent.start + min(tok_idxs), ent.start + max(tok_idxs) + 1, ent.label_)

class CRFTagger(Tagger):
    """ Conditional random field sequence tagger """
    def __init__(self, model_tag):
        log.info('Loading CRF sequence classifier: %s', model_tag)
        self.tagger = recognition.SequenceClassifier(model_tag)

    def tag(self, doc):
        offset = 0
        doc.tokens = []
        state = self.tagger.mapper.get_doc_state(doc)
        for sentence in self.tagger.mapper.iter_sequences(doc, state):
            for t in sentence:
                i = 0
                for i, c in enumerate(doc.text[t.idx:t.idx+len(t.text)]):
                    if c.isalnum():
                        break
                doc.tokens.append(Mention(t.idx+i, t.text))

            tags = self.tagger.tag(doc, sentence, state)
            start, tag_type = None, None
            for i, tag in enumerate(tags):
                if start != None and tag[0] != 'I':
                    yield self.mention_over_tokens(doc, start, i + offset, tag_type)
                    start, tag_type = None, None
                if tag[0] == 'B':
                    parts = tag.split('-')
                    if len(parts) == 2:
                        tag_type = parts[1]
                    start = i + offset
            if start != None:
                yield self.mention_over_tokens(doc, start, i + offset + 1, tag_type)
            offset += len(sentence)

class StanfordTagger(Tagger):
    """ Tag documents via a hosted Stanford NER service """
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def to_mention(self, doc, start, end):
        return mention

    @staticmethod
    def get_span_end(indexes, start, max_sz=1024):
        end = bisect_left(indexes, max_sz, lo=start)-1

        # if we can't find a step less than max_sz, we try to
        # take the smallest possible step and hope for the best
        if end <= start:
            end = start + 1

        return end

    def tag(self, doc):
        start_time = time()
        tokens = [t.text.replace('\n', ' ').replace('\r',' ') for t in doc.tokens]
 
        # the insanity below is motivated by the following:
        # - stanford doesn't tag everything we send it if we send it too much
        # - therefore we need to slice up the text into chunks of at most max_size
        # - however, we can't slice between sentences or tokens without hurting the tagger accuracy
        # - stanford doesn't return offsets, so we must keep tokens we send and tags returned aligned
        if tokens:
            acc = 0
            token_offsets = []
            character_to_token_offset = {0:-1}
            for i, t in enumerate(tokens):
                acc += len(t) + 1
                token_offsets.append(acc)
                character_to_token_offset[acc] = i

            # calculate character indexes of sentence delimiting tokens
            indexes = [0] + [token_offsets[i] for i,t in enumerate(tokens) if t == '.']
            if token_offsets and indexes[-1] != (token_offsets[-1]):
                indexes.append(token_offsets[-1])

            tags = []
            si, ei = 0, self.get_span_end(indexes, 0)
            while True:
                chunk_start_tok_idx = character_to_token_offset[indexes[si]]+1
                chunk_end_tok_idx = character_to_token_offset[indexes[ei]]+1
                text = ' '.join(tokens[chunk_start_tok_idx:chunk_end_tok_idx]).encode('utf-8')

                # todo: investigate why svc blows up if we don't RC after each chunk
                with tcp_socket(self.host, self.port) as s:
                    s.sendall(text+'\n')
                    buf = ''
                    while True:
                        buf += s.recv(4096)
                        if buf[-1] == '\n' or len(buf) > 10*len(text):
                            break
                    sentences = buf.split('\n')

                tags += [t.split('/')[-1] for s in sentences for t in s.split(' ')[:-1]]

                if ei+1 == len(indexes):
                    break
                else:
                    si, ei = ei, self.get_span_end(indexes, ei)

            if len(tags) != len(tokens):
                raise Exception('Tokenisation error: #tags != #tokens')

            start = None
            last = 'O'
            for i, (txt, tag) in enumerate(izip(tokens,tags)):
                if tag != last:
                    if last != 'O':
                        yield self.mention_over_tokens(doc, start, i)
                    last = tag
                    start = i
                i += 1

            if last != 'O':
                yield self.mention_over_tokens(doc, start, i)

        log.debug('Tagged doc (%s) with %i tokens in %.2fs', doc.id, len(doc.tokens), time() - start_time)
