import cPickle as pickle
import os
import tempfile
import pycrfsuite
from pymongo import MongoClient

from ..doc import Doc
from ..model.recognition import SequenceClassifier
from ..features import recognition

from nel import logging
log = logging.getLogger()

class TrainSequenceClassifier(object):
    """ Train a CRF sequence model for mention recognition over a corpus of documents. """
    def __init__(self, corpus, tag, classifier_id, nps_model_tag, verbose):
        if corpus == None:
            raise NotImplementedError    

        self.corpus_id = corpus
        self.tag_filter = tag
        self.classifier_id = classifier_id
        self.nps_model_tag = nps_model_tag
        self.verbose = verbose

    def __call__(self):
        log.info('Building training set...')
        docs = self.get_docs(self.corpus_id, self.tag_filter)

        log.info('Initialising feature extractors...')
        params = {
            'window': (-2, 2),
            'nps_model_tag': self.nps_model_tag
        }
        self.feature_extractor = recognition.SequenceFeatureExtractor(**params)

        trainer = pycrfsuite.Trainer(verbose=self.verbose)
        trainer.set_params({
            'c1': 1.,
            'c2': 0.001,
            'max_iterations': 200,
        })

        log.info('Extracting features for training instances...')
        for doc in docs:
            gold_mentions = sorted((m.begin,m.end,m.tag) for c in doc.chains for m in c.mentions)
            state = self.feature_extractor.get_doc_state(doc)
            for s in self.feature_extractor.iter_sequences(doc, state):
                features = self.feature_extractor.sequence_to_instance(doc, s, state)
                labels = list(self.iter_aligned_labels(s, gold_mentions))
                trainer.append(features, labels)

        model_path = os.path.join(tempfile.gettempdir(), tempfile.gettempprefix() + '.ner.model')

        log.info('Fitting model...')
        trainer.train(model_path)

        log.info('Loading temporarily serialized model into datastore: %s ...', model_path)
        data = None
        with open(model_path, 'r') as f:
            data = f.read()

        metadata = {
            'corpus': self.corpus_id,
            'tag': self.tag_filter
        }

        SequenceClassifier.create(self.classifier_id, data, params, metadata)
        log.info('Done.')

    @staticmethod
    def iter_aligned_labels(sequence, mentions):
        end = None
        tag = None
        next_start, next_end, next_tag = (None,None,None) if not mentions else mentions[0]
        for token in sequence:
            label = 'O'
            if next_start != None and (token.idx >= next_start or next_start < (token.idx+len(token.text))):
                end = next_end
                tag = next_tag
                label = 'B'
                if tag != None:
                    label += '-' + tag

                mentions.pop(0)
                next_start, next_end, next_tag = (None,None,None) if not mentions else mentions[0]
            elif end != None:
                if token.idx < end:
                    label = 'I'
                    if tag != None:
                        label += '-' + tag
                else:
                    end = None
            yield label

    @classmethod
    def get_docs(cls, corpus, tag):
        log.info('Fetching training docs (%s-%s)...', corpus or 'all', tag or 'all')
        store = MongoClient().docs[corpus]

        flt = {}
        if tag != None:
            flt['tag'] = tag

        # keeping all docs in memory could be problematic for large datasets
        # but simplifies computation of mapper parameters. todo: offline mapper prep
        return [Doc.obj(json) for json in store.find(flt)]

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('classifier_id', metavar='CLASSIFIER_ID')
        p.add_argument('nps_model_tag', metavar='NAME_PART_COUNT_MODEL')
        p.add_argument('--corpus', metavar='CORPUS', default=None, required=False)
        p.add_argument('--tag', metavar='TAG', default=None, required=False)
        p.add_argument('--verbose', default=False, action='store_true')
        p.set_defaults(cls=cls)
        return p
