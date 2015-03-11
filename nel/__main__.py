#!/usr/bin/env python
import argparse
import re
import sys
import textwrap
import logging

from corpora import prepare
from harness import harness
from features import extract
from learn import train
from model.prepare import wikilinks, wikipedia, kopi, derived, wordvec, dbpedia, yago, freebase

""" Logging Configuration """
logFormat = '%(asctime)s|%(levelname)s|%(module)s|%(message)s'
logging.basicConfig(format=logFormat)
log = logging.getLogger()
log.setLevel(logging.DEBUG)

APPS = [
    prepare.PrepareCorpus,
    harness.BatchLink,
    harness.ServiceHarness,
    extract.ExtractFeature,
    train.Train,

    derived.BuildLinkModels,
 
    wikilinks.BuildWikilinksLexicalisations,
    wikilinks.BuildWikilinksEntityContext,
    wikilinks.BuildWikilinksMentions,
    wikipedia.BuildWikipediaDocrep,
    wikipedia.BuildWikipediaRedirects,
    wikipedia.BuildWikiHitCounts,
    kopi.BuildKopiWikiEntityContext,
    
    derived.BuildEntitySet,
    derived.BuildIdfsForEntityContext,
    derived.BuildOccurrenceFromMentions,
    derived.BuildOccurrenceFromLinks,
    derived.BuildEntityCooccurrenceFromOccurrence,
    derived.BuildCandidateModel,
    
    wordvec.BuildWordVectors,
    wordvec.BuildEntityEmbeddings,
    
    dbpedia.BuildDbpediaLexicalisations,
    dbpedia.BuildDbpediaLinks,
    dbpedia.BuildDbpediaRedirects,

    yago.BuildYagoMeansNames, 
    freebase.BuildFreebaseCandidates,
]

def main(args=sys.argv[1:]):
    p = argparse.ArgumentParser(description='Named Entity Linker.')
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
