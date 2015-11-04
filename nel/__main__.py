#!/usr/bin/env python
import argparse
import re
import sys
import textwrap

from .corpora import prepare
from .harness import harness
from .features import extract
from .learn import ranking, resolving, recognition

import logging
log = logging.getLogger()

APPS = [
    prepare.PrepareCorpus,
    extract.ExtractFeature,
    recognition.TrainSequenceClassifier,
    ranking.TrainLinearRanker,
    resolving.TrainLinearResolver,
    resolving.FitNilThreshold,
    harness.BatchLink,
    harness.ServiceHarness
]

def main(args=sys.argv[1:]):
    p = argparse.ArgumentParser(description='nel entity linking framework')
    sp = p.add_subparsers()
    for cls in APPS:
        name = re.sub('([A-Z])', r'-\1', cls.__name__).lstrip('-').lower()
        help = cls.__doc__.split('\n')[0]
        desc = textwrap.dedent(cls.__doc__.rstrip())
        csp = sp.add_parser(name,
                            help=help,
                            description=desc,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
        cls.add_arguments(csp)
    namespace = vars(p.parse_args(args))
    cls = namespace.pop('cls')
    try:
        obj = cls(**namespace)
    except ValueError as e:
        p.error(str(e))
    obj()

if __name__ == '__main__':
    main()
