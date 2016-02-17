import json
from . import tag, candidates, coref, resolve
from ..features import feature

from nel import logging
log = logging.getLogger()

class Pipeline(object):
    """ Pipeline of document processes """
    def __init__(self, processors):
        self.processors = processors

    def __call__(self, doc):
        for c in self.processors:
            doc = c(doc)
        return doc

    @classmethod
    def load(cls, config_path):
        component_options = {
            component.__name__:{c.__name__:c for c in component.iter_options()}
            for component in [
                tag.Tagger,
                candidates.CandidateGenerator,
                coref.MentionClusterer,
                feature.Feature,
                resolve.Resolver
            ]
        }

        log.info('Loading pipeline configuration: %s ...', config_path)
        with open(config_path, 'r') as f:
            components = json.load(f)

        processors = []
        for item in components:
            if item['type'] not in component_options:
                raise Exception("Unknown component type '%s', select from: %s" % (item['type'], ', '.join(component_options.iterkeys())))

            options = component_options[item['type']]
            if item['name'] not in options:
                raise Exception("Unknown option '%s' for component type '%s', select from: %s" % (item['name'], item['type'], ', '.join(options.iterkeys())))

            processors.append(options[item['name']](**item['params']))

        return cls(processors)
