#!/usr/bin/env python
import argparse
import textwrap
import cPickle as pickle
import numpy

from pymongo import MongoClient
from time import time

from .feature import FEATURE_SET
from ..doc import Doc
from ..process.process import DocMapper

import logging
log = logging.getLogger()

class ExtractFeature(DocMapper):
    "Extract features from a prepared document model."
    FEATURES = FEATURE_SET
    def __init__(self, **kwargs):
        super(ExtractFeature, self).__init__(**kwargs)

        extractor_cls = kwargs.pop('featurecls')
        log.info("Preparing %s feature extractor...", extractor_cls.__name__)
        extractor_args = {p:kwargs[p] for p in extractor_cls.__init__.__code__.co_varnames if p in kwargs}
        self.mapper = extractor_cls(**extractor_args)

    @classmethod
    def add_arguments(cls, p):
        super(ExtractFeature, cls).add_arguments(p)

        sp = p.add_subparsers()
        for featurecls in cls.FEATURES:
            name = featurecls.__name__
            help_str = featurecls.__doc__.split('\n')[0]
            desc = textwrap.dedent(featurecls.__doc__.rstrip())
            csp = sp.add_parser(name,
                                help=help_str,
                                description=desc,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
            featurecls.add_arguments(csp)

        return p
