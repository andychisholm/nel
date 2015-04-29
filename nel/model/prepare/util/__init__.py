import re
import csv
import gzip
import logging

from collections import Counter
from glob import glob

log = logging.getLogger()

def iter_tsv(path, numfields):
    with open(path, 'r') as f:
        for parts in csv.reader(f, delimiter = '\t'):
            if len(parts) == numfields:
                yield [p.decode('utf-8') for p in parts]

def generic_read_open(path):
    log.debug('Reading file: ' + path)
    return gzip.open(path, 'rb') if path[-3:] == '.gz' else open(path, 'r')

def iter_fopens_at_path(path):
    for p in sorted(glob(path)):
        yield lambda: generic_read_open(p)

def iter_lines_from_files(path):
    for fopen in iter_fopens_at_path(path):
        with fopen() as f: 
            for l in f: yield l

# tokenisation and term counting
token_find_re = re.compile('\w+')
token_split_re = re.compile('\W+')
def iter_tokens(s): return token_find_re.finditer(s)
def tokenize(s):    return token_split_re.split(s)

ngram_vocab = None
def set_vocab(vocab):
    # todo: fix this dirty, dirty hack for parra
    global ngram_vocab
    ngram_vocab = vocab

def ngrams(tokens, n, vocab):
    num_tokens = len(tokens)
    for i in xrange(num_tokens):
        for j in xrange(i+1, min(num_tokens, i+n)+1):
            ngram = '_'.join(tokens[i:j])  # mikolov style
            if vocab == None or ngram in vocab:
                yield ngram
            else:
                # back off to a normalised version
                # might also try stemming
                normalised = ngram.lower()
                if normalised in vocab:
                    yield normalised

def term_counts(s, n, vocab):
    return Counter(ngrams(tokenize(s), n, vocab))

def trim_subsection_link(s):
    idx = s.find('#')
    return s if idx == -1 else s[:idx]

def normalise_wikipedia_link(s):
    s = s.replace(' ', '_').strip('_')
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s
