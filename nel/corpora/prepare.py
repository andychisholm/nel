import argparse
import textwrap
import numpy
from pymongo import MongoClient

from ..doc import Doc

from nel import logging
log = logging.getLogger()

class PrepareCorpus(object):
    """ Prepare and inject a corpus. """
    def __init__(self, **kwargs):
        self.corpus_id = kwargs.pop('corpus_id')
        self.parse = kwargs.pop('parsecls')(**kwargs)
        self.store = MongoClient()['docs'][self.corpus_id]

    def __call__(self):
        log.info("Dropping existing %s document...", self.corpus_id)
        self.store.drop()

        log.info('Loading documents for corpus %s...', self.corpus_id)
        for i, doc in enumerate(self.parse()):
            if i % 250 == 0:
                log.debug('Processed %i documents...', i)
            self.store.insert(doc.json())

        log.info('Import completed for %i documents.', i+1)

    APPS=set()
    @classmethod
    def Register(cls, c):
        cls.APPS.add(c)
        return c
    
    @classmethod
    def add_arguments(cls, p):
        p.add_argument('corpus_id', metavar='CORPUS_ID')
        p.set_defaults(cls=cls)
        
        app_name = 'prepare'
        sp = p.add_subparsers()
        for c in cls.APPS:
            name = c.__name__.lower()
            name = name.replace(app_name,'')
            csp = sp.add_parser(
                name,
                help=c.__doc__.split('\n')[0],
                description=textwrap.dedent(c.__doc__.rstrip()),
                formatter_class=argparse.RawDescriptionHelpFormatter)
            c.add_arguments(csp)
        return p
