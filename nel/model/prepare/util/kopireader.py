#!/usr/bin/env python
"""
Extract Wikipedia article-term count matrix from KOPI plain text dump.
"""
import glob
import gzip
import logging
import os
import re

log = logging.getLogger()

KOPI_DIR_FMT = 'wikipedia.txt.dump.{}-{}.SZTAKI'
KOPI_FN_FMT = '{}-wiki-{}_*.txt.gz'
ARTICLE_NAME_OPEN = '[['
ARTICLE_NAME_CLOSE = ']]'
EMPTY_NAME_LINE = '{}{}'.format(ARTICLE_NAME_OPEN, ARTICLE_NAME_CLOSE)
DISAMBIGUATION_CLOSE = ' (disambiguation)'
ENC = 'utf8'
REDIRECT_RE = r'#redirect'
NAMESPACES = [
    u"Talk",
    u"User",
    u"User talk",
    u"Wikipedia",
    u"WP", # Wikipedia alias
    u"Project", # Wikipedia alias
    u"Wikipedia talk",
    u"WT", # Wikipedia talk alias
    u"Project talk", # Wikipedia talk alias
    u"File",
    u"Image", # File alias
    u"File talk",
    u"Image talk", # File talk alias
    u"MediaWiki",
    u"MediaWiki talk",
    u"Template",
    u"Template talk",
    u"Help",
    u"Help talk",
    u"Category",
    u"Category talk",
    u"Portal",
    u"Portal talk",
    u"Book",
    u"Book talk",
    u"Draft",
    u"Draft talk",
    u"Education Program",
    u"Education Program talk",
    u"TimedText",
    u"TimedText talk",
    u"Module",
    u"Module talk",
    u"Special",
    u"Media",
    ]
PSEUDONAMESPACES = [
    u"CAT",
    u"H",
    u"P",
    u"T",
    ]
NAMESPACE_RE = r'(?:{}):'.format('|'.join(NAMESPACES))
PSEUDONAMESPACE_RE = r'(?:{}):'.format('|'.join(PSEUDONAMESPACES))

class KopiReader(object):
    "Read Wikipedia article text from KOPI dump."
    namespace_re = re.compile(NAMESPACE_RE, re.I)
    pseudonamespace_re = re.compile(PSEUDONAMESPACE_RE)
    redirect_re = re.compile(REDIRECT_RE, re.I)
    def __init__(self, root, date, lang):
        self.root = root
        self.date = date
        self.lang = lang

    def __str__(self):
        return 'KopiReader(root={},{}date={},{}lang={})'.format(
            self.root,
            '\n{:8s}'.format(''),
            self.date,
            '{:1s}'.format(''),
            self.lang
            )

    @property
    def path(self):
        "Path to directory containing KOPI plain text dump files."
        return os.path.join(
            self.root,
            KOPI_DIR_FMT.format(self.date, self.lang),
            KOPI_FN_FMT.format(self.date, self.lang)
            )

    def iter_files(self, partition = None, p_id = None):
        for i, f in enumerate(sorted(glob.glob(self.path))):
            if partition == None or (partition != None and i % partition == p_id):
                yield f

    def __call__(self):
        "Yield article text, saving names in same order."
        log.info('Reading Kopiwiki articles...')
        for f in self.iter_files():
            for name, text in self.read(f):
                yield (name, text)
        log.info('..done.')

    @staticmethod
    def read(path):
        "Yield (name,text) tuples from file."

        log.debug('Reading Kopiwiki file: %s' % os.path.split(path)[-1])
        
        name = KopiReader.get_name(EMPTY_NAME_LINE)
        lines = []
        redirect = False

        for line in gzip.open(path):
            line = line.decode(ENC).rstrip()
            if KopiReader.is_start(line):
                if KopiReader.is_article(name, redirect, lines):
                    yield KopiReader.format(name, lines)
                name = KopiReader.get_name(line)
                lines = []
                redirect = False
            elif KopiReader.is_redirect(line):
                redirect = True
            else:
                lines.append(line)
        if KopiReader.is_article(name, redirect, lines):
            yield KopiReader.format(name, lines)

    @staticmethod
    def get_name(line):
        "Return text portion of article name line."
        return line[2:-2]

    @staticmethod
    def is_redirect(line):
        "Return true if line indicates page is a redirect."
        return KopiReader.redirect_re.match(line)

    @staticmethod
    def is_start(line):
        "Return true if line indicates and article start."
        if not line.startswith(ARTICLE_NAME_OPEN):
            return False
        if not line.endswith(ARTICLE_NAME_CLOSE):
            return False
        return True

    @staticmethod
    def is_article(name, redirect, lines):
        "Return true if page is an article."
        if name == '':
            return False # empty name
        if redirect:
            return False # redirect page
        if name.endswith(DISAMBIGUATION_CLOSE):
            return False # disambiguation page
        if KopiReader.namespace_re.match(name):
            return False # not main article namespace
        if KopiReader.pseudonamespace_re.match(name):
            return False # not main article  namespace
        if len([l for l in lines if l != '']) == 0:
            return False # empty text
        return True

    @staticmethod
    def format(name, lines):
        "Format article as (name,text) tuple."
        return name, '\n'.join(lines)
