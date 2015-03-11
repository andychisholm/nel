# -*- coding: utf-8 -*-
from collections import namedtuple
from math import sqrt
import heapq


from numpy import concatenate, zeros

from numpy import array, uint32, uint8, random, empty, float32 as REAL
from six import iteritems, itervalues
from six.moves import xrange
from gensim.models.word2vec import Vocab


LabeledText = namedtuple('LabeledText', ['words', 'labels'])


# Generator that can pre-pad the sentences with nulls, like in the paper.
class LTIterator(object):
    def __init__(self, fname, min_count=0, mod_labels=False, only_return_starswith=None):
        self.fname = fname
        self.min_count = min_count
        self.fobj = open(self.fname)
        self.mod_labels = mod_labels
        self.only_return_starswith = only_return_starswith
        # self.break_out_at = 1000
        # self.loop_count = 0

    def __iter__(self):
        for line in self.fobj:
            # self.loop_count += 1
            # if self.loop_count > self.break_out_at:
            #     break
            l_text, l_label = ujson.loads(line[:-1] if line.endswith('\n') else line)
            if type(l_text) == str:
                l_text = l_text.split(' ')
            sentence_length = len(l_text)
            if self.min_count and sentence_length < self.min_count:
                l_text = (['null'] * (self.min_count - sentence_length)) + l_text
            if self.mod_labels and l_label and l_label[0][0] == 'α':
                if l_label[0].startswith('αþ'):
                    del l_label[0]
                    l_label.append('ζpos')
                elif l_label[0].startswith('αñ'):
                    del l_label[0]
                    l_label.append('ζneg')
            if not self.only_return_starswith:
                yield LabeledText(l_text, l_label)
            elif l_label[0].startswith(self.only_return_starswith):
                yield LabeledText(l_text, l_label)


def add_labeled_texts(doc2vec_obj, sentences):
    """
    Extends vocabulary labels from a sequence of sentences (can be a once-only generator stream).
    Each sentence must be a LabeledText-like object
    We don't want a new vocab, so we need something different from whats packaged w/ word2vec
    """
    orig_len_vocab = len(doc2vec_obj.vocab)
    orig_total_words = sum(v.count for v in itervalues(doc2vec_obj.vocab))
    threshold_count = float(doc2vec_obj.sample) * orig_total_words
    sentence_no, vocab = -1, {}
    rv_word_count = 1
    for sentence_no, sentence in enumerate(sentences):
        sentence_length = len(sentence.words)
        rv_word_count += int(rv_word_count * ((sqrt(rv_word_count / threshold_count) + 1) * (threshold_count / rv_word_count) if doc2vec_obj.sample else 1.0))
        for label in sentence.labels:
            if label not in doc2vec_obj.vocab:
                if label in vocab:
                    vocab[label].count += sentence_length
                else:
                    vocab[label] = Vocab(count=sentence_length)
    # assign a unique index to each new vocab item
    for word, v in iteritems(vocab):
        if v.count >= doc2vec_obj.min_count:
            v.index = len(doc2vec_obj.vocab)
            prob = (sqrt(v.count / threshold_count) + 1) * (threshold_count / v.count) if doc2vec_obj.sample else 1.0
            v.sample_probability = prob
            doc2vec_obj.index2word.append(word)
            doc2vec_obj.vocab[word] = v
    # add new vocab items to our hs tree
    if doc2vec_obj.hs:
        # doc2vec_obj.create_binary_tree()
        heap = list(itervalues(vocab))
        heapq.heapify(heap)
        for i in xrange(orig_len_vocab, len(doc2vec_obj.vocab) - 1):
            min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, Vocab(count=min1.count + min2.count, index=i + len(doc2vec_obj.vocab), left=min1, right=min2))
        if heap:
            max_depth, stack = 0, [(heap[0], [], [])]
            while stack:
                node, codes, points = stack.pop()
                if node.index < len(doc2vec_obj.vocab):
                    # leaf node => store its path from the root
                    node.code, node.point = codes, points
                    max_depth = max(len(codes), max_depth)
                else:
                    # inner node => continue recursion
                    points = array(list(points) + [node.index - len(doc2vec_obj.vocab)], dtype=uint32)
                    stack.append((node.left, array(list(codes) + [0], dtype=uint8), points))
                    stack.append((node.right, array(list(codes) + [1], dtype=uint8), points))
    # extend the model's vector sets
    random.seed(doc2vec_obj.seed)
    doc2vec_obj.syn0 = concatenate((doc2vec_obj.syn0, empty((len(doc2vec_obj.vocab) - orig_len_vocab, doc2vec_obj.layer1_size), dtype=REAL)))
    for i in xrange(orig_len_vocab, len(doc2vec_obj.vocab)):
        doc2vec_obj.syn0[i] = (random.rand(doc2vec_obj.layer1_size) - 0.5) / doc2vec_obj.layer1_size
    if doc2vec_obj.hs:
        doc2vec_obj.syn1 = concatenate((doc2vec_obj.syn1, zeros((len(doc2vec_obj.vocab) - orig_len_vocab, doc2vec_obj.layer1_size), dtype=REAL)))
    doc2vec_obj.syn0norm = None
    return rv_word_count

def train_model(model, data_csv, num_iters=1, min_count=0, label_indicators=None):
    if label_indicators is None:
        label_indicators = ['α', 'β', 'ζ']
    it = LTIterator(data_csv, min_count=min_count)
    model.build_vocab(it)
    m_total_wc = int(sum(v.count * v.sample_probability for k, v in iteritems(model.vocab) if not k[0] in label_indicators))
    for _ in xrange(num_iters):
        it = LTIterator(data_csv, min_count=min_count)
        model.train(it, total_words=m_total_wc)
