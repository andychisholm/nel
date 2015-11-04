""" Logging Configuration """
import logging

logFormat = '%(asctime)s|%(levelname)s|%(module)s|%(message)s'
logging.basicConfig(format=logFormat)
log = logging.getLogger()
log.setLevel(logging.DEBUG)

# register components via descriptors
from features import probability, context, meta, dummy, coherence, recognition # pylint: disable=I0011,W0611
from corpora import generic, conll, tac#pylint: disable=I0011,W0611
