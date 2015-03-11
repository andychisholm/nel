#!/usr/bin/env python
import re

from .process import Process
from ..doc import Mention

NON_WHITESPACE_RE = '\S+'
TOKEN_RE='([A-Za-z0-9_/\-&\']+|[?!.,-@])'
WORD_CHARS_RE = '\w+'
SINGLE_TAB_RE = '\\t'

class RegexTokeniser(Process):
    def __init__(self, regex=NON_WHITESPACE_RE):
        self.re = re.compile(regex)

    def __call__(self, doc):
        assert hasattr(doc, 'text'), 'doc must have text'
        doc.tokens = list(self._iter_tokens(doc))
        return doc

    def _iter_tokens(self, doc):
        for m in self.re.finditer(doc.text):
            yield Mention(m.start(), m.group())
