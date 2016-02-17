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

class CRFTagger(Tagger):
    """ Conditional random field sequence tagger """
    @classmethod
    def add_arguments(cls, p):
        p.add_argument('model_tag', metavar='MODEL_TAG')
        return p

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
    @classmethod
    def add_arguments(cls, p):
        p.add_argument('host', metavar='HOSTNAME')
        p.add_argument('port', type=int, metavar='PORT')
        return p

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

class SchwaTagger(Tagger):
    """ Tags named entities using the schwa docrep ner system (Dawborn, 15) """
    FLT_TAGS = ['date','cardinal','time','percent','ordinal','language','money','quantity']
    STARTUP_TIMEOUT = 30
    schema = None
 
    @classmethod
    def add_arguments(cls, p):
        p.add_argument('ner_package_path', metavar='NER_PACKAGE_PATH')
        p.add_argument('tagger_path', metavar='TAGGER_PATH')
        p.add_argument('tokenizer_path', metavar='TOKENIZER_PATH')
        p.add_argument('ner_model_name', metavar='NER_MODEL_NAME')
        return p

    def __init__(self, ner_package_path, tagger_path='schwa-ner-tagger', tokenizer_path='schwa-tokenizer', ner_model_name='conll12'):
        log.info('Initialising libschwa tagger...')
        self.tagger_path = tagger_path
        self.tokenizer_path = tokenizer_path
        self.ner_package_path = ner_package_path
        self.ner_model_name = ner_model_name

        if self.schema == None:
            # docrep schema used by the tokeniser and tagger
            from schwa import dr

            class Token(dr.Ann):
                raw = dr.Text()
                norm = dr.Text()
                pos = dr.Field()
                lemma = dr.Field()
                span = dr.Slice()
                tidx = dr.Field()

            class NamedEntity(dr.Ann):
                span = dr.Slice(Token)
                label = dr.Field()

            class SchwaDoc(dr.Doc):
                doc_id = dr.Field()
                tokens = dr.Store(Token)
                named_entities = dr.Store(NamedEntity)

            self.schema = SchwaDoc.schema()

        self.tagger_process = None
        self.initialise_tagger()

    def initialise_tagger(self):
        if self.tagger_process != None and self.tagger_process.poll() == None:
            self.tagger_process.kill()

        # configure the schwa ner tagger
        self.tagger_process = Popen([
            self.tagger_path,
            '--crf1-only', 'true',
            '--model', os.path.join(self.ner_package_path, self.ner_model_name)
        ],  cwd = self.ner_package_path, stdout = PIPE, stdin = PIPE, stderr = PIPE)

        t = time()
        while self.tagger_process.stderr.readline().strip() != 'Tagger Ready':
            if time() - t > self.STARTUP_TIMEOUT:
                raise Exception("Tagger startup timeout")
        log.info('Schwa tagger startup completed in %.1fs', time() - t)

    def text_to_dr(self, text):
        from schwa import dr
        tokenizer = Popen([
            self.tokenizer_path,
            '-p', 'docrep'
        ], cwd = self.ner_package_path, stdout = PIPE, stdin = PIPE)

        tok_dr, err = tokenizer.communicate(text)
        if not tok_dr or err:
            raise Exception("Schwa tokenizer failed while processing document")

        self.tagger_process.stdin.write(tok_dr)
        self.tagger_process.stdin.flush()

        try:
            status = self.tagger_process.stderr.readline().strip()
            if self.tagger_process.poll() != None or status == '' or status == '0':
                raise Exception("Schwa tagger failed while processing document")

            try:
                result_sz = int(status)
            except ValueError:
                schwa_error = Exception(status)
                raise Exception("Schwa tagger error while processing document", schwa_error)

            try:
                result_bytes = self.tagger_process.stdout.read(result_sz)
                result = dr.Reader(StringIO(result_bytes), self.schema).read()
                return result
            except Exception as e:
                raise Exception("Failed to deserialise schwa tagger output", e), None, sys.exc_info()[2]
        except:
            self.initialise_tagger()
            raise

    def tag(self, doc):
        raw = doc.text.encode('utf-8')

        # tagger returns byte offsets for tokens, we need unicode character offsets
        offset_map = byte_to_char_map(raw)
        tagged_dr = self.text_to_dr(raw)
        doc.tokens = [Mention(offset_map[t.span.start]-1, t.raw) for t in tagged_dr.tokens]

        for e in tagged_dr.named_entities:
            tag = e.label.lower()
            if tag not in self.FLT_TAGS:
                yield self.mention_over_tokens(doc, e.span.start, e.span.stop, tag)
