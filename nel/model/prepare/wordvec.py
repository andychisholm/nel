#!/usr/bin/env python -W ignore::DeprecationWarning
import gzip
import numpy
import os

from ..model import WordVectors
from ...util import parmapper

import logging
log = logging.getLogger()

class BuildWordVectors(object):
    "Build a word vector model from a binary word2vec file."
    def __init__(self, inpath, outpath):
        self.in_path = inpath
        self.out_path = outpath

    def __call__(self):
        self.build(self.in_path).write(self.out_path)

    def build(self, path):
        vocab = dict()

        with gzip.open(path) as f:
            vocab_sz, vec_sz = (int(c) for c in f.readline().split())
            
            log.debug('Reading %id word2vec model for %i words...' % (vec_sz, vocab_sz))

            vectors = numpy.empty((vocab_sz, vec_sz), dtype=numpy.float)
            vec_byte_len = numpy.dtype(numpy.float32).itemsize * vec_sz

            for line_idx in xrange(vocab_sz):
                word = ''
                while True:
                    ch = f.read(1)
                    if ch == ' ': break
                    if ch != '\n': word += ch

                vocab[word] = line_idx
                vectors[line_idx] = numpy.fromstring(f.read(vec_byte_len), numpy.float32)

        return WordVectors(vocab, vectors)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('inpath', metavar='INFILE')
        p.add_argument('outpath', metavar='OUTFILE')
        p.set_defaults(cls=cls)
        return p

from six import iteritems
from gensim.models.doc2vec import Doc2Vec
from collections import namedtuple
from schwa import dr
from time import time
VEC_SIZE = 100
WINDOW_SIZE = 10
MIN_COUNT = 9
ENTITY_PREFIX = '~'
LabeledText = namedtuple('LabeledText', ['words', 'labels'])

class Token(dr.Ann):
    norm = dr.Field()

class Section(dr.Ann):
    span = dr.Slice(Token)

class Doc(dr.Doc):
    name = dr.Field()
    tokens = dr.Store(Token)
    sections = dr.Store(Section)

class BuildEntityEmbeddings(object):
    """ Build entity embeddings from a document model. """
    def __init__(self, page_model_path, entity_model_path):
        self.page_model_path = page_model_path
        self.entity_model_path = entity_model_path
        #import marshal
        #self.entities = frozenset(marshal.load(open('/data0/linking/models/conll.entities.model', 'rb')))

    def __call__(self):
        log.info('Building corpus...')
       
        start = time()
        total = 4638072.0
        #self.corpus = list(self.iter_labeled_text())
        #for i, item in enumerate(self.iter_labeled_text()):
        #    self.corpus.append(item)
        #    if i % 50000 == 0:
        #        rate = (i+1) /  (time() - start)
        #        eta = ((total - i)/rate)/60
        #        log.info('Processed %i docs (%.2f), rate: %.1f/s, eta: %.1f mins', i, i*100.0/total, rate, eta)

        model_dm = Doc2Vec(sentences=None, size=VEC_SIZE, window=WINDOW_SIZE, min_count=MIN_COUNT, workers=30, dm=1, sample=1e-3)
        log.info("Building vocab...")
        model_dm.build_vocab(self.iter_labeled_text())
        dm_total_wc = int(sum(v.count * v.sample_probability for k, v in iteritems(model_dm.vocab) if k[0] != ENTITY_PREFIX))

        num_iters = 1 
        for i in xrange(0, num_iters):
            log.info("Training iteration %d/%d...", i+1, num_iters)
            log.warn("DEBUG: %.4f", model_dm.similarity('~Apple_Inc.', '~Samsung_Electronics'))
            log.warn(str(model_dm.most_similar('~Apple_Inc.')))
            #model_dm.alpha = 0.025 * (num_iters - i) / num_iters + 0.0001 * i / num_iters
            #model_dm.min_alpha = model_dm.alpha
            model_dm.train(self.iter_labeled_text(), total_words=dm_total_wc)

        model_dm.train_words = False
        model_dm.save(self.entity_model_path)
        import code
        code.interact(local=locals())

    def labeled_text_for_file(self, path):
        log.info('Processing entity embeddings: %s...', path)
        
        instances = []
        with open(path,'rb')  as f:
            reader = dr.Reader(f, Doc.schema())
            for doc in reader:
                #if doc.name not in self.entities:
                #    continue
                tokens = [t.norm for t in doc.tokens[doc.sections[0].span]] + [t.norm for s in doc.sections[1:] for t in doc.tokens[s.span][:100]]
                #tokens = [t.norm for t in doc.tokens[doc.sections[0].span]]
                #tokens = [t.norm for t in doc.tokens[:500]]
                if MIN_COUNT and len(tokens) < MIN_COUNT:
                    tokens = (['__NULL__'] * (MIN_COUNT - len(doc.tokens))) + tokens

                instances.append(LabeledText(tokens, [ENTITY_PREFIX + doc.name]))

        return instances

    def iter_file_names(self):
        for path, _, files in os.walk(self.page_model_path):
            for filename in files:
                if filename.endswith('.dr'):
                    yield os.path.join(path, filename)

    def iter_labeled_text(self):
        log.info('Processing test doc embeddings...')
        import cPickle as pickle
        doc_model_paths = [
            '/data0/linking/data/conll/prepared/train.doc.model',
            '/data0/linking/data/conll/prepared/testa.doc.model',
            '/data0/linking/data/conll/prepared/testb.doc.model'
        ]
        for path in doc_model_paths:
            f = open(path, 'rb')
            log.info('Processing: %s ...', path)
            while True:
                try:
                    doc = pickle.load(f)
                    yield LabeledText([t.text for t in doc.tokens],['#~' + doc.id])
                except EOFError:
                    break

        with parmapper(self.labeled_text_for_file, 30, recycle_interval=None) as pm:
            for _, instances in pm.consume(self.iter_file_names()):
                for x in instances:
                    yield x

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('page_model_path', metavar='DOC_MODEL')
        p.add_argument('entity_model_path', metavar='OUT_PATH')
        p.set_defaults(cls=cls)
        return p
