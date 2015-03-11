#!/usr/bin/env python
from codecs import getreader
from rdflib.parser import create_input_source
from rdflib.plugins.parsers.ntriples import NTriplesParser
from rdflib.plugins.parsers.ntriples import ParseError
from rdflib.plugins.parsers.ntriples import r_tail
from rdflib.plugins.parsers.ntriples import r_wspace

class NtStream(NTriplesParser):
    def iter(self, inputsource):
        """Iter f as an n-triples file."""
        inputsource = create_input_source(source=inputsource,
                                          format='ntriples')
        source = inputsource.getByteStream()
        if not hasattr(source, 'read'):
            raise ParseError("Item to parse must be a file-like object.")
        source = getreader('utf-8')(source)
        self.file = source
        self.buffer = ''
        while True:
            self.line = __line = self.readline()
            if self.line is None:
                break
            self.eat(r_wspace)
            if (not self.line) or self.line.startswith(('#')):
                continue  # The line is empty or a comment
            try:
                yield self.parseline()
            except ParseError as msg:
                raise ParseError("Invalid line (%s):\n%r" % (msg, __line))

    def parseline(self):
        subject = self.subject()
        self.eat(r_wspace)
        predicate = self.predicate()
        self.eat(r_wspace)
        obj = self.object()
        self.eat(r_tail)
        if self.line:
            raise ParseError("Trailing garbage")
        return subject, predicate, obj
