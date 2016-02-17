# -*- coding: utf-8 -*-
import argparse
import textwrap
import numpy

from collections import defaultdict
from pymongo import MongoClient

from ..doc import Doc

from nel import logging
log = logging.getLogger()

class CorpusStats(object):
    """ Output summary statistics for documents in a corpus """
    def __init__(self, **kwargs):
        self.corpus_id = kwargs.pop('corpus_id')
        self.store = MongoClient()['docs'][self.corpus_id]

    def __call__(self):
        total_non_nil = 0
        total_candidate_recalled = 0
        chain_mention_counts = []
        chain_candidate_counts = []

        entities = set()
        candidates = set()

        num_docs = 0
        doc_tags = defaultdict(int)

        coref_errors = 0
        coref_eval_chains = 0

        log.info('Loading documents for corpus %s...', self.corpus_id)
        for i, raw_doc in enumerate(self.store.find()):
            doc = Doc.obj(raw_doc)
            num_docs += 1
            doc_tags[doc.tag] += 1
            if i % 250 == 0:
                log.debug('Processed %i documents...', i)
            
            if doc.tag != 'train':
                for chain in doc.chains:
                    coref_eval_chains += 1
                    chain_res = chain.mentions[0].resolution
                    chain_res = chain_res.id if chain_res else None
                    for m in chain.mentions:
                        mention_res = m.resolution.id if m.resolution else None
                        if mention_res != chain_res:
                            coref_errors += 1
                            break

            # accumulate mention statistics
            for chain in doc.chains:
                chain_mention_counts.append(len(chain.mentions))
                chain_candidate_counts.append(len(chain.candidates))
                for c in chain.candidates:
                    candidates.add(c.id)

                for m in chain.mentions:
                    if m.resolution != None:
                        entities.add(m.resolution.id)
                        total_non_nil += 1
                        if m.resolution.id in [c.id for c in chain.candidates]:
                            total_candidate_recalled += 1
                    #else:
                    #    log.warn('No candidate for chain: %s - %s', chain.resolution.id, ', '.join(set("'" + m.text.lower() + "'" for m in chain.mentions)))

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
        log.info('Coref P                  = %.2f', coref_errors / float(coref_eval_chains))

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
 
    @classmethod
    def add_arguments(cls, p):
        p.add_argument('corpus_id', metavar='CORPUS_ID')
        p.set_defaults(cls=cls)
        return p
