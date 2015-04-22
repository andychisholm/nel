#!/usr/bin/env python
import os
import re
import six
import datetime
import bz2
import HTMLParser
import urllib

from time import time
from schwa import dr, tokenizer
from ...util import parmapper
from ..model import Redirects

import logging
log = logging.getLogger()

class BuildWikipediaRedirects(object):
    """ Build redirect model from comprssed wikipedia-dump """
    def __init__(self, wikipedia_dump_path, model_tag):
        self.wikipedia_dump_path = wikipedia_dump_path
        self.model_tag = model_tag
        self.html_parser = HTMLParser.HTMLParser()

    def __call__(self):
        log.info('Building redirect model from: %s' % self.wikipedia_dump_path)
        Redirects(self.model_tag).create(self.iter_mappings())
        log.info('Done.')

    def iter_mappings(self):
        """ Based on http://code.google.com/p/wikipedia-redirect/ """
        WK_TITLE_LINE_BEGIN = "    <title>"
        WK_TITLE_LINE_END = "</title>\n"
        WK_REDIRECT_LINE = "    <redirect"
        WK_TEXT_LINE = "      <text xml"

        title_line = None
        redirect = False
        with bz2.BZ2File(self.wikipedia_dump_path, 'r') as f:
            for line in f:
                line = line.decode('utf-8')
                if line.startswith(WK_TITLE_LINE_BEGIN):
                    title_line = line
                    redirect = False
                elif line.startswith(WK_REDIRECT_LINE):
                    redirect = True
                elif redirect and line.startswith(WK_TEXT_LINE):
                    start = line.find('[[', len(WK_TEXT_LINE))
                    end = line.find(']]', start)
                    target = self.normalise(line[start+2:end])
                    source = self.normalise(title_line[len(WK_TITLE_LINE_BEGIN):-len(WK_TITLE_LINE_END)])
                    redirect = False
                    if self.valid_source(source):
                        yield source, target

    def valid_source(self, source):
        return not source.startswith("Wikipedia:") and \
               not source.startswith("Template:") and \
               not source.startswith("Portal:") and \
               not source.startswith("List of ")

    def normalise(self, title):
        return self.html_parser.unescape(title).replace(' ', '_')

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('wikipedia_dump_path', metavar='WIKIDUMP_PATH')
        p.add_argument('--model_tag', default='wikipedia', required=False, metavar='REDIRECT_MODEL_TAG')
        p.set_defaults(cls=cls)
        return p

class BuildWikiHitCounts(object):
    "Build mapping from wikipedia name to page view count from raw wikipedia hit files."
    def __init__(self, hitcounts_path, entities_model_path):
        self.in_path = hitcounts_path
        self.entities_model_path = entities_model_path

    def __call__(self):
        self.build().write(self.models_path)

    def build(self):
        hcs = HitCount()

        log.info('Parsing page view files for %i entities...' % len(self.entities.d))

        for line in iter_lines_from_files(self.in_path + '*.gz'):
            try:
                if line[:3] == 'en ':
                    page, count = line.split()[1:3]
                    page = urllib.unquote(page).strip().decode('utf-8')

                    # the simplest way of determining whether a hit url refers to 
                    # an actual article is to compare it with a known set of article titles
                    # (also restricts our count model to only those entities we actually care about)
                    if self.entities.contains_wkid(page):
                        hcs.update(page, hcs.count(page) + int(count))

            except: continue

        log.info('Built model with counts for %i entities' % hcs.size())

        return hcs

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('indir', metavar='INDIR')
        p.add_argument('modeldir', metavar='MODELDIR')
        p.set_defaults(cls=cls)
        return p

class Token(dr.Ann):
    span = dr.Slice()
    raw = dr.Field()
    norm = dr.Field()

class Link(dr.Ann):
    span = dr.Slice(Token)
    anchor = dr.Field()
    target = dr.Field()

class Section(dr.Ann):
    title = dr.Field()
    span =  dr.Slice(Token)

class Sentence(dr.Ann):
    span = dr.Slice(Token)

class Paragraph(dr.Ann):
    span = dr.Slice(Sentence)

class WikiDoc(dr.Doc):
    id = dr.Field()
    name = dr.Field()
    tokens = dr.Store(Token)
    links = dr.Store(Link)
    sections = dr.Store(Section)
    sentences = dr.Store(Sentence)
    paragraphs = dr.Store(Paragraph)

""" Tokenizer Callback Interface Methods """
class FragmentTokenizer(object):
    def __init__(self):
        self.tokenizer = tokenizer.Tokenizer()
        self.clear_doc()

    def start_doc(self, id, title):
        self.doc = WikiDoc()
        self.offset = 0
        self.doc.id = id
        self.doc.name = title.replace(' ', '_')
        self.start_section(title)

    def clear_doc(self):
        self.doc = None
        self.offset = 0
        self.section_start = None

    def finalise_doc(self):
        d = self.doc
        self.finalise_section()
        self.clear_doc()
        return d

    def start_section(self, title):
        self.finalise_section()
        self.doc.sections.append(Section(title=title))
        self.section_start = self.len()

    def finalise_section(self):
        if self.section_start != None:
            self.doc.sections[-1].span = slice(self.section_start, self.len())
        self.section_start = None

    def append_fragment(self, text, link_spans):
        assert isinstance(text, six.text_type)
        offset = len(self.doc.tokens)
        self.tokenizer.tokenize(text.encode(ENC), dest=self)
        self.mark_between(text.encode(ENC), self.doc.tokens[offset:], offset, link_spans)

    def mark_between(self, text, tokens, offset, links):
        prev_stop = 0

        current_link_tok_start = None
        current_link = None

        for i, tok in enumerate(tokens):
            tok.before = text[prev_stop:tok.span.start]
            prev_stop = tok.span.stop

            if not current_link and links:
                if tok.span.start >= links[0][0][0]: # ugh
                    current_link = links.pop(0)
                    current_link_tok_start = i

            if current_link:
                if tok.span.stop >= current_link[0][1]:
                    _, anchor, target = current_link

                    self.doc.links.append(Link(
                        span=slice(offset+current_link_tok_start, offset+i+1),
                        anchor=anchor,
                        target=target))

                    #raw = text[text_span[0]:text_span[1]]
                    #tok_span_str = '[' + tokens[current_link_tok_start].norm.encode('utf-8') + ']' + ''.join('[' + t.before + t.norm.encode('utf-8') + ']' for t in tokens[current_link_tok_start+1:i+1])
                    #log.debug('LINK OVER: (%s) (%s)', raw, tok_span_str)

                    current_link_tok_start = None
                    current_link = None

        try:
            tok.after = text[prev_stop:]
        except NameError: 
            pass # No tokens

    def unhandled(self, method_name, *args):
        log.info('%r unhandled during tokenization (args=%s)', method_name, args)

    def error(self, start, raw):
        log.error('Error processing %r at %d', raw, start)

    def add(self, start, raw, norm=None):
        span = slice(self.offset + start, self.offset + start + len(raw))
        norm = self.fix_token_norm(norm or raw).decode(ENC, errors='ignore')

        try:
            raw.decode(ENC)
        except:
            pass
            #log.warn('Decode error: %s', raw)

        raw = raw.decode(ENC, errors='ignore')
        tok = Token(span=span, raw=raw if norm != raw else None, norm=norm)
        self.doc.tokens.append(tok)

    def fix_token_norm(self, norm):
        return norm

    def len(self, annot='tokens'):
        return len(getattr(self.doc, annot))
    
    def begin_sentence(self):
        self.sent_start = self.len()

    def end_sentence(self):
        self.doc.sentences.append(Sentence(span=slice(self.sent_start, self.len())))
        delattr(self, 'sent_start')

    def begin_paragraph(self):
        self.para_start = self.len('sentences')

    def end_paragraph(self):
        self.doc.paragraphs.append(Paragraph(span=slice(self.para_start, self.len('sentences'))))
        delattr(self, 'para_start')

    # tokeniser interprets dashes at the start of a sentence as list items
    # so these callbacks are sporadically called 
    def begin_list(self): pass
    def end_list(self): pass
    def begin_item(self): pass
    def end_item(self): pass

    def begin_heading(self, h): pass
    def end_heading(self): pass
    def begin_document(self): pass
    def end_document(self): pass

ENC='utf-8'
class BuildWikipediaDocrep(object):
    """ Convert plaintext wikipedia corpus to docrep """
    DOC_START_RE = re.compile(r'<doc id="(?P<id>[0-9]+)" url="(?P<url>.+)" title="(?P<title>.+)">')
    DOC_END_RE = re.compile(r'</doc>')
    SECTION_HEAD_RE = re.compile(r'<h(?P<level>[0-9])>(?P<title>.*)</h[0-9]>')
    LIST_DELIN_ITEM_RE = re.compile(r'</?ul>')
    LIST_ITEM_RE = re.compile(r'<li>(?P<content>.*)</li>')
    LINK_RE = re.compile(r'<a href="(?P<target>.+?)">(?P<anchor>.+?)</a>')

    def __init__(self, wikipedia_data_path, outpath):
        self.in_path = wikipedia_data_path
        self.out_path = outpath

        self.tokenizer = FragmentTokenizer()
        self.doc_count = 0

    def docs_for_file(self, fn):
        return list(self.iter_docs_for_file(fn))

    def iter_docs_for_file(self, fn):
        with bz2.BZ2File(fn,'r') as f:
            expected_title_line = None
            for line in f:
                # dodgy removal of redundant section titles embedded as text
                if expected_title_line and expected_title_line == line.strip().rstrip('.'):
                    expected_title_line = None
                    continue
                else:
                    expected_title_line = None

                if line.strip() == '':
                    continue

                if not self.tokenizer.doc:
                    # we're between documents elements
                    m = self.DOC_START_RE.match(line)
                    if m:
                        self.tokenizer.start_doc(m.group('id'), m.group('title').decode(ENC))
                        expected_title_line = m.group('title')
                        #log.info('DOC_START: %s', m.group('title'))
                else:
                    # parsing a document element
                    if self.DOC_END_RE.match(line):
                        # end of current document
                        if self.tokenizer.doc:
                            #log.info('DOC_END')
                            yield self.tokenizer.finalise_doc()
                            self.doc_count += 1
                        else:
                            log.error('Unmatched doc end element.')
                    else:
                        if self.LIST_DELIN_ITEM_RE.match(line):
                            continue
                        m = self.SECTION_HEAD_RE.match(line)
                        if m:
                            # note that title may be empty
                            #log.info('SECTION: %s', m.group('title'))
                            self.tokenizer.start_section(m.group('title'))
                            expected_title_line = m.group('title')
                            # todo: add section annotation
                        else:
                            m = self.LIST_ITEM_RE.match(line)
                            if m:
                                line = m.group('content')

                            if line.strip() == '':
                                continue

                            # find links
                            links = []
                            trim_offset = 0
                            for link in self.LINK_RE.finditer(line):
                                anchor = link.group('anchor').decode(ENC)
                                target = link.group('target').decode(ENC).replace(' ', '_')
                                span = (link.start()-trim_offset, link.start()+len(link.group('anchor')) - trim_offset)

                                #log.debug('LINK: %s -> %s', anchor, target)

                                links.append((span, anchor, target))
                                trim_offset += link.end() - link.start() - len(link.group('anchor'))

                            # if we found links, remove them from the raw text
                            if links:
                                line = self.LINK_RE.sub(r'\2', line)

                            #log.debug('TEXT: %s', line)
                            self.tokenizer.append_fragment(line.decode(ENC), links)

    def iter_file_names(self):
        for path, _, files in os.walk(self.in_path):
            for filename in files:
                if filename.startswith('wiki_') and filename.endswith('.bz2'):
                    yield os.path.join(path, filename)

    def iter_doc_reps(self):
        # we can get a little speedup from parallelisation but not much (io?)
        with parmapper(self.docs_for_file, 4, recycle_interval=None) as pm:
            for _, docs in pm.consume(self.iter_file_names()):
                for d in docs: yield d

        #for fn in self.iter_file_names():
        #    for doc in self.iter_docs_for_file(fn):
        #        yield doc

    def __call__(self):
        WK_PAGES_EST = 4630000

        with open(self.out_path,'w') as f:
            i = 0
            writer = dr.Writer(f, WikiDoc)
            try:
                log.info('Processing docs...')
                start_time = time()
                for i, doc in enumerate(self.iter_doc_reps()):
                    if i == 10000 or (i % 100000 == 0 and i > 0):
                        dps = (i+1)/float(time() - start_time)
                        eta = datetime.timedelta(seconds=int(WK_PAGES_EST / dps))
                        log.info('Processed %i documents... %.2f d/s (eta: %s)', i, dps, eta) 
                    writer.write(doc)
            except:
                log.error('Failed on doc: %i', i)
                raise
    @classmethod
    def add_arguments(cls, p):
        p.add_argument('wikipedia_data_path', metavar='WIKI_DATA_PATH')
        p.add_argument('outpath', metavar='OUTFILE')
        p.set_defaults(cls=cls)
        return p
