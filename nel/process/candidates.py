import re
import string
from itertools import izip

from .process import Process
from ..model import recognition
from ..doc import Mention, Chain, Candidate
from ..model.disambiguation import NameProbability

from nel import logging
log = logging.getLogger()

class CandidateGenerator(Process):
    """ Populate candidate sets for chains in a document. """
    RE_WS = re.compile('\s+')
    RE_WS_PRE_PUCT = re.compile(u'\s+([^a-zA-Z\d])')

    @classmethod
    def iter_options(cls):
        for c in globals().itervalues():
            if c != cls and isinstance(c, type) and issubclass(c, cls):
                yield c

    def get_doc_state(self, doc):
        return None

    def get_candidates(self, doc, name, state):
        raise NotImplementedError

    def __call__(self, doc):
        state = self.get_doc_state(doc)

        for chain in doc.chains:
            forms = set()
            for m in chain.mentions:
                forms.update(self.get_normalised_forms(m.text))
            forms = sorted(forms, key=len, reverse=True)

            candidates = set()
            for sf in forms:
                candidates.update(self.get_candidates(doc, chain, sf, state))
                if candidates:
                    break

            chain.candidates = [Candidate(c) for c in candidates]

        return doc

    @classmethod
    def normalise_form(cls, sf):
        sf = sf.lower()
        sf = cls.RE_WS_PRE_PUCT.sub(r'\1', sf)
        sf = cls.RE_WS.sub(' ', sf)
        return sf

    @classmethod
    def iter_derived_forms(cls, sf):
        yield sf
        yield sf.replace("'s", "")
        yield ''.join(c for c in sf if not c in string.punctuation)

        comma_parts = sf.split(',')[:-1]
        for i in xrange(len(comma_parts)):
            yield ''.join(comma_parts[:i+1])
        if comma_parts:
            yield ''.join(comma_parts)
        
        colon_idx = sf.find(':')
        if colon_idx != -1:
            yield sf[:colon_idx]

        quote_parts = sf.split('"')
        if len(quote_parts) >= 3:
            yield ''.join(quote_parts)
            yield ''.join(quote_parts[:1]+quote_parts[-1:])

    @classmethod
    def get_normalised_forms(cls, sf):
        return set(cls.normalise_form(f) for f in cls.iter_derived_forms(sf))

class NameCounts(CandidateGenerator):
    """ Return candidates for entities with non-zero name posterior. """
    def __init__(self, name_model_tag, limit):
        log.info('Preparing name model candidate generator (model=%s, limit=%i)...', name_model_tag, limit)
        self.nm = NameProbability(name_model_tag)
        self.limit = limit
 
    @classmethod
    def add_arguments(cls, p):
        p.add_argument('name_model_tag', metavar='NAME_MODEL')
        p.add_argument('--limit', required=False, type=int, default=15, metavar='MAX_CANDIDATES')
        return p

    def get_doc_state(self, doc):
        forms = list(set(df for c in doc.chains for m in c.mentions for df in self.get_normalised_forms(m.text)))

        state = {}
        for sf, eps in self.nm.get_counts_for_names(forms).iteritems():
            state[sf] = [e for e, c in sorted(eps.iteritems(), key=lambda (k,v):v, reverse=True)][:self.limit]
        return state

    def get_candidates(self, doc, chain, name, state):
        return state[name][:self.limit]
