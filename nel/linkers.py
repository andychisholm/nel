#!/usr/bin/env python
import json
import numpy
import math

from time import time
from .process.resolve import GreedyOverlapResolver
from .process.tag import StanfordTagger, CandidateGenerator
from .process.tokenise import RegexTokeniser, TOKEN_RE
from .process.resolve import FeatureRankResolver
from .process.process import Pipeline
from .features.feature import FEATURE_SET
from .features.meta import ClassifierScore

import logging
log = logging.getLogger()

class OnlineLinker(Pipeline):
    """ Supervised linker computing document features on demand """
    def __init__(self, config_path):
        log.info('Loading linker configuration: %s ...', config_path)
        self.config = json.load(open(config_path,'r'))

        log.info('Initialising feature extractors...')
        feature_clss = {f.__name__:f for f in FEATURE_SET}
        invalid_features = set(f['name'] for f in self.config['features'] if f['name'] not in feature_clss)
        if invalid_features:
            raise Exception("Invalid feature class: (%s)" % ', '.join(invalid_features))

        feature_extractors = [feature_clss[f['name']](**f['params']) for f in self.config['features']]

        resolution = RankingResolver(**self.config['resolver'])

        log.info('Building pipeline...')
        self.processors = [
            RegexTokeniser(TOKEN_RE), 
            StanfordTagger(**self.config['tagger']['params']),
            CandidateGenerator(**self.config['candidate_generation']['params']),
        ] + feature_extractors \
          + resolution.processors

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('config_path', metavar='FEATURE_EXTRACTOR_CONFIG_PATH')
        p.set_defaults(linkcls=cls)
        return p

class RankingResolver(Pipeline):
    """ Resolves mentions by sorting candidates on a specified features. """
    def __init__(self, ranker, resolver = None):
        self.processors = [
            FeatureRankResolver(ranker, resolver),
            GreedyOverlapResolver(ranker)
        ]
    
    @classmethod
    def add_arguments(cls, p):
        p.add_argument('ranker', metavar='RANKING_FEATURE_ID', help='Id of the feature used to rank candidates')
        p.add_argument('resolver', required=False, default=None, metavar='RESOLVING_FEATURE_ID', help='Id of the feature used to resolve candidates')
        return p

LINKERS = set([OnlineLinker])
