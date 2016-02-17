import re
import string
import math
from itertools import chain
from spacy.en import English

from ..model.recognition import NamePartCounts

from nel import logging
log = logging.getLogger()

class SequenceFeatureExtractor(object):
    def __init__(self, **kwargs):
        self.nlp = English()
        self.window = kwargs.pop('window')
        self.nps_model_tag = kwargs.pop('nps_model_tag')

        log.info('Preparing sequence feature extractor... (window=%s)', str(self.window))
        self.window_feature_extractors = [
            WordFeatures(),
            TagFeatures(),
        ]
        self.feature_extractors = [
            GazeteerFeatures(self.nps_model_tag)
        ]

    def get_doc_state(self, doc):
        shared_state = self.nlp(doc.text)
        extractor_states = {f:f.get_doc_state(doc, shared_state) for f in self.feature_extractors}
        return (shared_state, extractor_states)

    def iter_sequences(self, doc, state):
        shared_state, extractor_states = state
        for sentence in shared_state.sents:
            yield list(sentence)

    def sequence_to_instance(self, doc, tokens, state):
        shared_state, extractor_states = state
        instance = []
        features = [
            {f:v for fe in self.window_feature_extractors for f,v in fe(doc, t)}
        for t in tokens]

        for i in xrange(len(tokens)):
            token_features = {
                'bias': 1.0
            }
            for w in xrange(self.window[0], self.window[1]+1):
                j = i + w
                if j < -1 or j > len(tokens):
                    continue
                if j == -1:
                    token_features['BOS'] = 1.0
                elif j == len(tokens):
                    token_features['EOS'] = 1.0
                else:
                    token_features[str(w)] = features[j]
            instance.append(token_features)

        for fe in self.feature_extractors:
            for i, t in enumerate(tokens):
                instance[i][fe.__class__.__name__] = dict(fe(doc, t, extractor_states[fe]))

        return instance

class GazeteerFeatures(object):
    def __init__(self, np_model_tag):
        self.np_model = NamePartCounts(np_model_tag)

    def to_feature(self, count, total):
        if total > 0 and count > 0:
            p = count / float(total)
        else:
            p = 0.
        return p

    def iter_nps_features(self, nps, i):
        O = nps.get('O', 0)
        B,I,E = nps.get('B', 0), nps.get('I', 0), nps.get('E', 0)
        if B and i <= 0:
            yield 'word.NP[%i]_B'%i, self.to_feature(B, O)
            yield 'word.N[%i]_B'%i, math.log(B)
        if I:
            yield 'word.NP[%i]_I'%i, self.to_feature(I, O)
            yield 'word.N[%i]_I'%i, math.log(I)
        if E and i >= 0:
            yield 'word.NP[%i]_E'%i, self.to_feature(E, O)
            yield 'word.N[%i]_E'%i, math.log(E)
        if O and i == 0:
            yield 'word.N[%i]_O'%i, math.log(O)

    def get_doc_state(self, doc, shared_state):
        terms = set(t.text for t in shared_state)
        for i in xrange(len(shared_state)):
            terms.add(shared_state[i:i+1].text)
            terms.add(shared_state[i:i+2].text)
        return self.np_model.get_part_counts(terms)

    def __call__(self, doc, token, state):
        tfs = [self.iter_nps_features(state[token.text], 0)]

        doc = token.doc
        if token.i < len(doc)-1:
            nps = state[doc[token.i:token.i+2].text]
            tfs.append(self.iter_nps_features(nps, 1))
        if token.i > 0:
            nps = state[doc[token.i-1:token.i+1].text]
            tfs.append(self.iter_nps_features(nps, -1))

        return chain.from_iterable(tfs)

class WordFeatures(object):
    @staticmethod
    def to_word_pattern(word):
        word = re.sub('[A-Z]', 'A', word)
        word = re.sub('[a-z]', 'a', word)
        word = re.sub('[0-9]', '0', word)
        return word

    @staticmethod
    def reduce_word_pattern(wp):
        wp = re.sub('A+', 'A', wp)
        wp = re.sub('a+', 'a', wp)
        wp = re.sub('0+', '0', wp)
        return wp

    def __call__(self, doc, token):
        word = token.text
        word_pattern = self.to_word_pattern(word)
        word_pattern_red = self.reduce_word_pattern(word_pattern)
        yield 'word.lower=' + word.lower(), 1.
        yield 'word.isupper=%s' % word.isupper(), 1.
        yield 'word.wp=' + word_pattern, 1.
        yield 'word.wpr=' + word_pattern_red, 1.
        yield 'word.istitle=%s' % word.istitle(), 1.
        yield 'word.allupper=%s' % (word.upper() == word), 1.
        yield 'word.alllower=%s' % (word.lower() == word), 1.
        yield 'word.isdigit=%s' % word.isdigit(), 1.
        yield 'word.subcaps=%s' % any(c.isupper() for c in word[1:]), 1.
        yield 'word.haspunct=%s' % any(c in string.punctuation for c in word), 1.
        yield 'word.hasdigits=%s' % any(c in string.digits for c in word), 1.
        yield 'word.apos=%s' % word.endswith("'s"), 1.
        yield 'word[-3:]=' + word[-3:], 1.
        yield 'word[-2:]=' + word[-2:], 1.
        yield 'word[:2]=' + word[:2], 1.
        yield 'word[:3]=' + word[:3], 1.
        yield 'word.raw[-1]=' + doc.raw[token.idx], 1.

        end_offset = token.idx+len(token.text)+1
        yield 'word.raw[+1]=' + (doc.raw[end_offset] if end_offset < len(doc.text) else 'EOD'), 1.

class TagFeatures(object):
    def __call__(self, doc, token):
        yield 'word.postag=' + token.tag_, 1.
        yield 'word.pos=' + token.pos_, 1.
        yield 'word.ent=' + token.ent_iob_, 1.
        yield 'word.enttype=' + token.ent_type_, 1.
        yield 'word.dep=' + token.dep_, 1.
        yield 'word.pos[:2]=' + token.tag_[:2], 1.

        #lead_offset = 0
        #for c in doc.text[token.idx:token.idx+len(token.text)]:
        #    if c != ' ':
        #        break
        #    lead_offset += 1

        #import code
        #code.interact(local=locals())
