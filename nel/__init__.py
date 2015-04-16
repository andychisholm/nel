__pkg_name__ = 'nel'
__version__ = '0.1'

""" Logging Configuration """
import logging

logFormat = '%(asctime)s|%(levelname)s|%(module)s|%(message)s'
logging.basicConfig(format=logFormat)
log = logging.getLogger()
log.setLevel(logging.DEBUG)

# register components via descriptors
from features import probability, context, meta, dummy # pylint: disable=I0011,W0611
from corpora import conll #pylint: disable=I0011,W0611
