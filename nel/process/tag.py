#!/usr/bin/env python
import os

from itertools import izip
from time import time
from bisect import bisect_left

from .process import Process
from ..model import model
from ..doc import Mention, Chain, Candidate
from ..util import group, spanset_insert, tcp_socket

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
        assert hasattr(doc, 'tokens'), 'doc must have tokens'
        doc.chains = self.cluster_mentions(self._tag(doc))

    def _tag(self, doc):
        """Yield mention annotations."""
        raise NotImplementedError

    def _mention_over_tokens(self, doc, i, j):
        """Create mention annotation from token i to token j-1."""
        toks = doc.tokens[i:j]
        begin = toks[0].begin
        end = toks[-1].end
        text = doc.text[begin:end]

        return Mention(begin, text)

class CandidateGenerator(Process):
    def __init__(self, name_model_path):
        self.candidates = model.Candidates()

        n = model.Name(lower=True)
        n.read(name_model_path)
        self.entities_by_name =  {k:v.keys() for k, v in n.d.iteritems()}
    
    def __call__(self, doc):
        for chain in doc.chains:
            forms = sorted(set(m.text for m in chain.mentions),key=len,reverse=True)
            candidates = []
            for sf in forms:
                candidates = set(self.candidates.search(sf)).union(
                             set(self.entities_by_name.get(sf.lower(), [])))
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

    def _tag(self, doc):
        start_time = time()
        i = 0
        start = None
        last = 'O'

        tokens = [t.text.replace('\n', ' ').replace('\r',' ') for t in doc.tokens]

        # calculate sentence offsets 
        indexes = [-1] + [i for i,t in enumerate(tokens) if t == '.']
        if indexes[-1] != (len(tokens)-1):
            indexes.append(len(tokens)-1)

        tags = []
        max_sz = 2000
        s = 0
        e = bisect_left(indexes, max_sz, lo=s)-1
        while True:
            text = ' '.join(tokens[indexes[s]+1:indexes[e]+1]).encode('utf-8')
            
            # todo: investigate why svc blows up if we don't RC after each chunk
            with tcp_socket(self.host, self.port) as s:
                s.sendall(text+'\n')
                tagged = s.recv(10*len(text))
                sentences = tagged.split('\n')
            tags += [t.split('/')[-1] for s in sentences for t in s.split(' ')[:-1]]
            
            if e+1 == len(indexes):
                break
            else:
                s = e
                e = bisect_left(indexes, indexes[s]+max_sz, lo=s)-1

        if len(tags) != len(tokens):
            raise Exception('Tokenisation error')
        for i, (txt, tag) in enumerate(izip(tokens,tags)):
            if tag != last:
                if last != 'O':
                    yield self._mention_over_tokens(doc, start, i)
                last = tag
                start = i
            i += 1

        if last != 'O':
            yield self._mention_over_tokens(doc, start, i)

        log.debug('Tagged doc (%s) with %i tokens in %.1fs', doc.id, len(doc.tokens), time() - start_time)

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
