#!/usr/bin/env python
import argparse
import re
import sys
import textwrap

from .corpora import prepare, analysis, visualise
from .harness import harness
from .learn import ranking, resolving, recognition

from .process.process import CorpusProcessor
from .process.tag import Tagger
from .process.candidates import CandidateGenerator
from .process.coref import MentionClusterer
from .features.feature import Feature

from nel import logging
log = logging.getLogger()

APPS = [
    prepare.PrepareCorpus,
    analysis.CorpusStats,
    visualise.CompareCorpusAnnotations,
    recognition.TrainSequenceClassifier,
    ranking.TrainLinearRanker,
    resolving.TrainLinearResolver,
    resolving.FitNilThreshold,
    harness.BatchLink,
    harness.ServiceHarness
]

CORPUS_PROCESSORS = [
    ('tag-documents', Tagger),
    ('generate-candidates', CandidateGenerator),
    ('cluster-mentions', MentionClusterer),
    ('extract-feature', Feature),
]

def add_subparser(sp, cls, name = None, doc_text = None):
    name = name or cls.__name__
    doc_text = doc_text or cls.__doc__
    csp = sp.add_parser(
        name,
        help=doc_text.split('\n')[0],
        description=textwrap.dedent(doc_text.rstrip()),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    return csp

def main(args=sys.argv[1:]):
    p = argparse.ArgumentParser(description='nel entity linking framework')
    sp = p.add_subparsers()

    for cls in APPS:
        csp = add_subparser(sp, cls, name=re.sub('([A-Z])', r'-\1', cls.__name__).lstrip('-').lower())
        cls.add_arguments(csp)

    for name, cls in CORPUS_PROCESSORS:
        csp = add_subparser(sp, CorpusProcessor, name=name, doc_text=cls.__doc__)
        CorpusProcessor.add_arguments(csp)
        subsp = csp.add_subparsers()
        for subcls in cls.iter_options():
            subcsp = add_subparser(subsp, subcls)
            subcls.add_arguments(subcsp)
            subcsp.set_defaults(mappercls=subcls)

    namespace = vars(p.parse_args(args))
    cls = namespace.pop('cls')
    try:
        obj = cls(**namespace)
    except ValueError as e:
        p.error(str(e))
    obj()

if __name__ == '__main__':
    main()
