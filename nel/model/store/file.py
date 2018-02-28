import os
import mmap
import cPickle as pickle
import operator
from functools32 import lru_cache

from ..data import Store, ObjectStore
from nel import logging
log = logging.getLogger()

class FileObjectStore(ObjectStore):
    def __init__(self, path):
        self.path = path
        self.store = mmdict(path)

    @classmethod
    def get_protocol(cls):
        return 'file'

    def iter_ids(self):
        return self.store.iterkeys()

    def exists(self, oid):
        return oid in self.store

    def fetch(self, oid):
        return self.store[oid]

    def fetch_many(self, oids):
        return [self.fetch(oid) for oid in oids]

    def fetch_all(self):
        return self.store.iteritems()

    def save_many(self, obj_iter):
        self.store.close()
        mmdict.write(self.path, ((o['_id'], o) for o in obj_iter))
        self.store = mmdict(self.path)

    def save(self,obj):
        self.save_many([obj])
       
    @classmethod
    def GetPath(cls, store_id, uri):
        path = store_id.replace(':', '/')
        if uri and uri.startswith('file://'):
            path = os.path.join(uri[7:], path)
        return path

    @classmethod
    def Get(cls, store_id, uri='file://', **kwargs):
        return cls(cls.GetPath(store_id, uri))

class mmdict(object):
    def __init__(self, path):
        self.path = path
        self.index = {}
        
        index_path = self.path + '.index'
        if os.path.exists(index_path):
            log.debug('Loading mmap store: %s ...' % index_path)
            with open(index_path, 'rb') as f:
                self.index = dict(self.deserialise(f))

            self.data_file = open(path + '.data', 'rb')
            self.data_mmap = mmap.mmap(self.data_file.fileno(), 0, prot=mmap.PROT_READ)
        else:
            log.warn('No existing mmap store found: %s ...' % index_path)

    @staticmethod
    def serialise(obj, f):
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def deserialise(f):
        return pickle.load(f)

    @staticmethod
    def static_itervalues(path):
        with open(path + '.data', 'rb') as f:
            while True:
                try:
                    yield mmdict.deserialise(f)
                except EOFError: break

    def iteritems(self):
        sorted_idx = sorted(self.index.iteritems(), key=operator.itemgetter(1))

        for i, v in enumerate(self.itervalues()):
            yield (sorted_idx[i][0], v)

    def iterkeys(self):
        return self.index.iterkeys()

    def itervalues(self):
        self.data_mmap.seek(0)
        while True:
            try:
                yield self.deserialise(self.data_mmap)
            except EOFError: break

    def __len__(self):
        return len(self.index)

    def __contains__(self, key):
        return key in self.index

    @lru_cache(maxsize=20000)
    def __getitem__(self, key):
        if key not in self:
            return None

        self.data_mmap.seek(self.index[key])
        return self.deserialise(self.data_mmap)

    def __enter__(self):
        return self

    def close(self):
        if hasattr(self, 'data_mmap') and self.data_mmap != None:
            self.data_mmap.close()
        if hasattr(self, 'data_file') and self.data_file != None:
            self.data_file.close()

    def __exit__(self, type, value, traceback):
        self.close()

    def __del__(self):
        self.close()

    @staticmethod
    def write(path, iter_kvs):
        index = []
        with open(path + '.data', 'wb') as f:
            for key, value in iter_kvs:
                index.append((key, f.tell()))
                mmdict.serialise(value, f)
        with open(path + '.index','wb') as f:
            mmdict.serialise(index, f)
