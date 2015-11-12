# -*- coding: utf-8 -*-
import argparse
import textwrap
import numpy

from collections import defaultdict
from pymongo import MongoClient

import logging
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

        # mention statistics
        total_non_nil = 0
        total_candidate_recalled = 0
        chain_mention_counts = []
        chain_candidate_counts = []

        entities = set()
        candidates = set()

        num_docs = 0
        doc_tags = defaultdict(int)

        log.info('Loading documents for corpus %s...', self.corpus_id)
        for i, doc in enumerate(self.parse()):
            num_docs += 1
            doc_tags[doc.tag] += 1
            if i % 100 == 0:
                log.debug('Processed %i documents...', i)

            # accumulate mention statistics
            for chain in doc.chains:
                chain_mention_counts.append(len(chain.mentions))
                chain_candidate_counts.append(len(chain.candidates))
                for c in chain.candidates:
                    candidates.add(c.id)
                if chain.resolution != None:
                    entities.add(chain.resolution.id)
                    total_non_nil += len(chain.mentions)
                    if chain.resolution.id in [c.id for c in chain.candidates]:
                        total_candidate_recalled += len(chain.mentions)
                    else:
                        log.warn('No candidate for chain: %s - %s', chain.resolution.id, ', '.join(set("'" + m.text.lower() + "'" for m in chain.mentions)))

            self.store.insert(doc.json())

        total_chains = len(chain_mention_counts)
        total_mentions = sum(chain_mention_counts)

        section_delimiter = '-' * 40
        log.info(section_delimiter)
        log.info('CORPUS STATISTICS')
        log.info(section_delimiter)
        log.info('Docs                     = %i', num_docs)
        for tag,count in doc_tags.iteritems():
            log.info('\t%.1f%% %s (%i)', float(count*100) / num_docs, tag, count)

        log.info(section_delimiter)
        log.info('Total mentions           = %i', total_mentions)
        log.info('Total nil mentions (%%)   = %i (%.2f)', total_mentions - total_non_nil, float(total_mentions - total_non_nil) / total_mentions)

        log.info(section_delimiter)
        log.info('Total chains             = %i', total_chains)
        log.info('Mentions per Chain (σ)   = %.1f (%.2f)', numpy.mean(chain_mention_counts), numpy.std(chain_mention_counts))

        log.info(section_delimiter)
        log.info('Total entities           = %i', len(entities))
        log.info('Total candidates         = %i', len(candidates))
        log.info(section_delimiter)
        log.info('Candidates per Chain (σ) = %.1f (%.2f)', numpy.mean(chain_candidate_counts), numpy.std(chain_candidate_counts))

        no_candidates_count = sum(1 for c in chain_candidate_counts if c == 0)
        candidate_recall = 'n/a' if total_non_nil == 0 else '%.2f' % (float(total_candidate_recalled) / total_non_nil)
        log.info('Candidate Recall (%%)     = %s', candidate_recall)
        log.info('Nil Candidate Chains (%%) = %i (%.2f)', no_candidates_count, float(no_candidates_count) / total_chains)
        log.info(section_delimiter)

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
