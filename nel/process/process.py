#!/usr/bin/env python
class Process(object):
    def __call__(self, doc):
        """Add annotations to doc and return it"""
        raise NotImplementedError

class Pipeline(object):
    def __init__(self, processors):
        self.processors = processors

    def __call__(self, doc):
        for c in self.processors:
            doc = c(doc)
        return doc
