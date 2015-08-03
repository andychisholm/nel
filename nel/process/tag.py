#!/usr/bin/env python
import os

from itertools import izip
from time import time
from bisect import bisect_left
from schwa import dr
from subprocess import Popen, PIPE
from threading import Thread

from .process import Process
from ..model import model
from ..doc import Mention, Chain, Candidate
from ..util import group, spanset_insert, tcp_socket, byte_to_char_map, StreamingQueue

import logging
log = logging.getLogger()

class Tagger(Process):
    """ Tags and performs naive coref over mentions in tokenised text. """
    @staticmethod
    def cluster_mentions(mentions): 
        chains = []
        unchained_mentions = sorted(mentions, key=lambda m:m.begin, reverse=True)

        #log.debug('MENTIONS: ' + ';'.join(m.text for m in unchained_mentions))
        while unchained_mentions:
            mention = unchained_mentions.pop(0)

            potential_antecedents = [(m.text, m) for m in unchained_mentions]
            chain = [mention]

            likely_acronym = False

            if mention.text.upper() == mention.text:
                # check if our mention is an acronym of a previous mention
                for a, m in potential_antecedents:
                    if (''.join(p[0] for p in a.split(' ') if p).upper() == mention.text) or \
                       (''.join(p[0] for p in a.split(' ') if p and p[0].isupper()).upper() == mention.text):
                        chain.insert(0, m)
                        unchained_mentions.remove(m)
                        likely_acronym = True
                potential_antecedents = [(m.text, m) for m in unchained_mentions]

            last = None
            longest_mention = mention
            while last != longest_mention and potential_antecedents:
                # check if we are a prefix/suffix of a preceding mention
                n = longest_mention.text.lower()
                for a, m in potential_antecedents:
                    na = a.lower()
                    if (likely_acronym and mention.text == a) or \
                       (not likely_acronym and (na.startswith(n) or na.endswith(n) or n.startswith(na) or n.endswith(na))):
                        chain.insert(0, m)
                        unchained_mentions.remove(m)
                
                last = longest_mention
                longest_mention = sorted(chain, key=lambda m: len(m.text), reverse=True)[0]
                potential_antecedents = [(m.text, m) for m in unchained_mentions]

            chains.append(Chain(mentions=chain))
            #log.debug('CHAIN(%i): %s' % (len(potential_antecedents), ';'.join(m.text for m in chain)))
        
        return chains

    def __call__(self, doc):
        doc.chains = self.cluster_mentions(self._tag(doc))
        return doc

    def _tag(self, doc):
        """Yield mention annotations."""
        raise NotImplementedError

    def _mention_over_tokens(self, doc, i, j, tag=None):
        """Create mention annotation from token i to token j-1."""
        toks = doc.tokens[i:j]
        begin = toks[0].begin
        end = toks[-1].end
        text = doc.text[begin:end]

        return Mention(begin, text, tag)

class CandidateGenerator(Process):
    def __init__(self, candidate_model_tag):
        self.candidates = model.Candidates(candidate_model_tag)
 
    def __call__(self, doc):
        for chain in doc.chains:
            forms = sorted(set(m.text for m in chain.mentions),key=len,reverse=True)
            candidates = []
            for sf in forms:
                candidates = self.candidates.search(sf)
                if candidates:
                    break
            chain.candidates = [Candidate(c) for c in candidates]
        return doc

class StanfordTagger(Tagger):
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

    def _tag(self, doc):
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
                        yield self._mention_over_tokens(doc, start, i)
                    last = tag
                    start = i
                i += 1

            if last != 'O':
                yield self._mention_over_tokens(doc, start, i)

        log.debug('Tagged doc (%s) with %i tokens in %.2fs', doc.id, len(doc.tokens), time() - start_time)

# docrep schema used by the tokeniser and tagger
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

class SchwaTagger(Tagger):
    """ Tags named entities using the schwa docrep ner system (Dawborn, 15) """
    FLT_TAGS = ['date','cardinal','time','percent','ordinal','language','money','quantity']
    def __init__(self, ner_package_path, tagger_path='schwa-ner-tagger', tokenizer_path='schwa-tokenizer', ner_model_name='conll12', dequeue_timeout=0.001):
        self.tagger_path = tagger_path
        self.tokenizer_path = tokenizer_path
        self.ner_package_path = ner_package_path
        self.ner_model_name = ner_model_name
        self.dequeue_timeout = dequeue_timeout
        self.schema = SchwaDoc.schema()
        self.tagger_process = None
        self.initialise_tagger()

    def initialise_tagger(self):
        if self.tagger_process != None and self.tagger_process.poll() == None:
            # this should indirectly terminate the thread listing on stdout also
            self.tagger_process.kill()

        # configure the schwa ner tagger
        self.tagger_process = Popen([
            self.tagger_path,
            '--crf1-only', 'true',
            '--model', os.path.join(self.ner_package_path, self.ner_model_name)
        ],  cwd = self.ner_package_path, stdout = PIPE, stdin = PIPE)

        # setup a thread to listen for output from the tagger process
        # we need this to facilitate async reads from stdout
        self.out_queue = StreamingQueue(timeout=self.dequeue_timeout)
        self.enqueue_thread = Thread(target=self.enqueue_process_output, args=(self.tagger_process, self.out_queue))
        self.enqueue_thread.daemon = True
        self.enqueue_thread.start()

        # output stream reader
        self.out_queue_reader = dr.Reader(self.out_queue, self.schema)

    @staticmethod
    def enqueue_process_output(process, queue):
        while process.poll() == None:
            queue.put(process.stdout.read(1))

    def text_to_dr(self, text):
        tokenizer = Popen([
            self.tokenizer_path,
            '-p', 'docrep'
        ], cwd = self.ner_package_path, stdout = PIPE, stdin = PIPE)

        tok_dr, _ = tokenizer.communicate(text)
        self.tagger_process.stdin.write(tok_dr)
        self.tagger_process.stdin.flush()
        try:
            result = self.out_queue_reader.read()
            if self.tagger_process.poll() != None:
                # if the tagger process goes down here, something's gone wrong and the result is probably None
                raise Exception("Tagger failed while processing document")
            return result
        except:
            self.initialise_tagger()
            raise

    def normalise_text(self, text):
        # here we map funny unicode character into their ascii equivalents so as not to offend the tagger
        text = text.replace(u'\xa2','c')
        text = text.replace(u'\u2014','-')
        text = text.replace(u'\u2018',"'").replace(u'\u2019',"'")
        text = text.replace(u'\u201c','"').replace(u'\u201d','"')
        return text

    def _tag(self, doc):
        encoding = 'utf-8'
        raw = self.normalise_text(doc.text).encode(encoding)

        # tagger returns byte offsets for tokens, we need unicode character offsets
        offset_map = byte_to_char_map(raw)
        tagged_dr = self.text_to_dr(raw)
        doc.tokens = [Mention(offset_map[t.span.start]-1, t.raw) for t in tagged_dr.tokens]

        for e in tagged_dr.named_entities:
            tag = e.label.lower()
            if tag not in self.FLT_TAGS:
                yield self._mention_over_tokens(doc, e.span.start, e.span.stop, tag)

MAX_MENTION_LEN = 4
class LookupTagger(Tagger):
    """ Create mentions over all ngrams in a document which match items in an alias set. """
    def __init__(self, max_len=MAX_MENTION_LEN):
        import nltk
        from ..util import trie
        self.max_len = max_len

    def _tag(self, doc):
        """Yield mention annotations."""
        tagged_spans = []
        for m in doc.mentions:
            spanset_insert(tagged_spans, m.begin, m.end)
            yield m

        tags = [tag for _, tag in nltk.pos_tag([t.text for t in doc.tokens])]
        num_tokens = len(doc.tokens)
        i = 0
        while i < num_tokens:
            j = i + self.max_len
            while j > i:
                if tags[i][0] == 'N':
                    mention = self._mention_over_tokens(doc, i, j)
                    candidates = self.candidates.search(mention.text)
                    if candidates and spanset_insert(tagged_spans, mention.begin, mention.end):
                        mention.candidates = [Candidate(e) for e in candidates]
                        yield mention
                        i = j-1
                        break
                j -= 1
            i += 1
