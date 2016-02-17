import os
import redis
import re
import ujson as json

from collections import defaultdict
from itertools import islice, izip
from pymongo import MongoClient

from nel import logging
log = logging.getLogger()

DATASTORE_URI_VAR = 'NEL_DATASTORE_URI'
DEFAULT_DATASTORE_URI = 'redis://localhost'

class Store(object):
    def flush(self):
        raise NotImplementedError

    def exists(self, oid):
        raise NotImplementedError

    def delete(self, oid):
        raise NotImplementedError

    def iter_ids(self):
        raise NotImplementedError

    @classmethod
    def get_protocol(cls):
        raise NotImplementedError

class StoreBase(object):
    @classmethod
    def Get(cls, store_id, **kwargs):
        uri = kwargs.pop('uri', None)
        uri = uri or os.environ.get(DATASTORE_URI_VAR, DEFAULT_DATASTORE_URI)
        proto = uri.split(':')[0] if uri else None

        # lookup is cached to avoid reflection
        if not hasattr(cls, '_impl_by_proto'):
            cls._impl_by_proto = {c.get_protocol():c for c in cls.__subclasses__()}

        store_cls = cls._impl_by_proto.get(proto, None)

        if store_cls == None:
            log.error('Unsupported data store proto (%s), choose from (%s)', proto, ','.join(cls._impl_by_proto.iterkeys()))
            raise NotImplementedError

        fmt_cls_name = re.sub('([A-Z])', r' \1', store_cls.__name__).strip().lower()
        log.debug("Using %s for (%s)...", fmt_cls_name, store_id)

        return store_cls.Get(store_id, uri, **kwargs)

class ObjectStore(StoreBase):
    def fetch(self, oid):
        raise NotImplementedError

    def save(self, obj):
        raise NotImplementedError

    def save_many(self, obj_iter):
        raise NotImplementedError

    def fetch_all(self):
        raise NotImplementedError

    def fetch_many(self, oids):
        raise NotImplementedError

    def batched_inserter(self, batch_size):
        return BatchInserter(self, batch_size)

class FieldStore(StoreBase):
    def fetch(self, oid):
        raise NotImplementedError

    def fetch_all(self):
        raise NotImplementedError

    def fetch_many(self, oids):
        raise NotImplementedError

    def fetch_field(self, oid, field):
        raise NotImplementedError

    def inc(self, oid, field, value):
        raise NotImplementedError

    def inc_many(self, inc_op_iter):
        raise NotImplementedError

class SetStore(StoreBase):
    def add(self, oid, item):
        self.union(oid, [item])

    def union(self, oid, items):
        raise NotImplementedError

    def add_many(self, oid_item_iter):
        raise NotImplementedError

    def fetch(self, oid):
        raise NotImplementedError

    def fetch_many(self, oid_iter):
        raise NotImplementedError

class BatchedOperation(object):
    def __init__(self, operation, batch_size):
        self.batch_size = batch_size
        self.batch = []
        self.operation = operation

    def flush(self):
        if self.batch:
            self.operation(self.batch)
            self.batch = []

    def append(self, obj):
        self.batch.append(obj)
        if len(self.batch) >= self.batch_size:
            self.flush()

    def __enter__(self):
        self.batch = []
        return self

    def __exit__(self, etype, evalue, etraceback):
        if etype == None:
            self.flush()

class BatchInserter(BatchedOperation):
    def __init__(self, store, batch_size):
        super(BatchInserter, self).__init__(store.save_many, batch_size)

import os
import pkgutil
store_cls_path = [os.path.join(os.path.dirname(__file__), "store")]
for loader, name, _ in pkgutil.walk_packages(store_cls_path):
    __import__('nel.model.store', fromlist=[name])
