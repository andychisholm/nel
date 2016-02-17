from collections import defaultdict
from itertools import chain

from ..doc import Chain
from .process import Process

from nel import logging
log = logging.getLogger()

class MentionClusterer(Process):
    """ Perform in-document coreference clustering over mentions """
    @classmethod
    def iter_options(cls):
        for c in globals().itervalues():
            if c != cls and isinstance(c, type) and issubclass(c, cls):
                yield c

    def get_clusters(self, doc, mentions):
        raise NotImplementedError

    def __call__(self, doc):
        mentions_to_cluster = []

        # for training documents, we use gold resolutions to cluster non-nil mentions
        if doc.tag == 'train':
            mbr = defaultdict(list)
            for c in doc.chains:
                for m in c.mentions:
                    if m.resolution == None:
                        mentions_to_cluster.append(m)
                    else:
                        mbr[m.resolution.id].append(m)
            doc.chains = [Chain(mentions=ms) for r, ms in mbr.iteritems()]
        else:
            mentions_to_cluster = list(chain.from_iterable(c.mentions for c in doc.chains))
            doc.chains = []

        doc.chains += [Chain(mentions=ms) for ms in self.get_clusters(doc, mentions_to_cluster)]
        return doc

class SpanOverlap(MentionClusterer):
    """ Cluster mentions by prefix/suffix overlap and acronym match. """
    @classmethod
    def add_arguments(cls, p):
        return p

    def __init__(self):
        log.info('Using mention text span-overlap coreference clusterer...')

    def get_clusters(self, doc, mentions):
        chains = []
        unchained_mentions = sorted(mentions, key=lambda m:m.begin, reverse=True)

        #log.debug('MENTIONS: ' + ';'.join(m.text for m in unchained_mentions))
        while unchained_mentions:
            mention = unchained_mentions.pop(0)

            potential_antecedents = [(m.text, m) for m in unchained_mentions if m.tag == mention.tag]
            chain = [mention]

            likely_acronym = False

            if mention.text.upper() == mention.text:
                # check if our mention is an acronym of a previous mention
                for a, m in potential_antecedents:
                    if (''.join(p[0] for p in a.split(' ') if p).upper() == mention.text) or \
                       (''.join(p[0] for p in a.split(' ') if p and p[0].isupper()).upper() == mention.text):
                        chain.insert(0, m)
                        unchained_mentions.remove(m)
                        likely_acronym = True
                potential_antecedents = [(m.text, m) for m in unchained_mentions]

            last = None
            longest_mention = mention
            while last != longest_mention and potential_antecedents:
                # check if we are a prefix/suffix of a preceding mention
                n = longest_mention.text.lower()
                for a, m in potential_antecedents:
                    na = a.lower()
                    if (likely_acronym and mention.text == a) or \
                       (not likely_acronym and (na.startswith(n) or na.endswith(n) or n.startswith(na) or n.endswith(na))):
                        chain.insert(0, m)
                        unchained_mentions.remove(m)
                
                last = longest_mention
                longest_mention = sorted(chain, key=lambda m: len(m.text), reverse=True)[0]
                potential_antecedents = [(m.text, m) for m in unchained_mentions if m.tag == mention.tag]

            chains.append(chain)
            #log.debug('CHAIN(%i): %s' % (len(potential_antecedents), ';'.join(m.text for m in chain)))

        return chains
