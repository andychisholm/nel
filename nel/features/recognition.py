import re
import string
from spacy.en import English

import logging
log = logging.getLogger()

class SequenceFeatureExtractor(object):
    def __init__(self, **kwargs):
        self.nlp = English()
        self.window = kwargs.pop('window', (-2, 2))

        log.info('Preparing sequence feature extractor... (window=%s)', str(self.window))
        self.feature_extractors = [f() for f in [
            WordFeatures,
            TagFeatures
        ]]

    def iter_sequences(self, doc):
        for sentence in self.nlp(doc.text).sents:
            yield list(sentence)

    def sequence_to_instance(self, doc, tokens):
        instance = []
        features = [[f for fe in self.feature_extractors for f in fe(doc, t)] for t in tokens]
        for i in xrange(len(features)):
            instance.append(['bias'])
            for w in xrange(self.window[0], self.window[1]+1):
                j = i + w
                if j < -1 or j > len(tokens):
                    continue
                if j == -1:
                    instance[i].append('BOS')
                elif j == len(tokens):
                    instance[i].append('EOS')
                else:
                    instance[i].extend(u'{:d}:{:s}'.format(w, f) for f in features[j])
        return instance

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
        yield 'word.lower=' + word.lower()
        yield 'word[-3:]=' + word[-3:]
        yield 'word[-2:]=' + word[-2:]
        yield 'word[:2]=' + word[:2]
        yield 'word[:3]=' + word[:3]
        yield 'word.istitle=%s' % word.istitle()
        yield 'word.isupper=%s' % word.isupper()
        yield 'word.allupper=%s' % (word.upper() == word)
        yield 'word.alllower=%s' % (word.lower() == word)
        yield 'word.isdigit=%s' % word.isdigit()
        yield 'word.subcaps=%s' % any(c.isupper() for c in word[1:])
        yield 'word.haspunct=%s' % any(c in string.punctuation for c in word)
        yield 'word.hasdigits=%s' % any(c in string.digits for c in word)
        yield 'word.apos=%s' % word.endswith("'s")
        yield 'word.wp=' + word_pattern
        yield 'word.wpr=' + word_pattern_red
        yield 'word.raw[-1]=' + doc.raw[token.idx]

        end_offset = token.idx+len(token.text)+1
        yield 'word.raw[+1]=' + (doc.raw[end_offset] if end_offset < len(doc.text) else 'EOD')

class TagFeatures(object):
    def __call__(self, doc, token):
        yield 'word.pos=' + token.tag_
        yield 'word.ent=' + token.ent_iob_
        yield 'word.dep=' + token.dep_
        yield 'word.pos[:2]=' + token.tag_[:2]

        #lead_offset = 0
        #for c in doc.text[token.idx:token.idx+len(token.text)]:
        #    if c != ' ':
        #        break
        #    lead_offset += 1

        #import code
        #code.interact(local=locals())
