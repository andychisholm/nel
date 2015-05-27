import os
import redis
import re
import ujson as json

from collections import defaultdict
from itertools import islice, izip
from pymongo import MongoClient

import logging
log = logging.getLogger()

DATASTORE_URI_VAR = 'NEL_DATASTORE_URI'

class Store(object):
    def fetch(self, oid):
        raise NotImplementedError

    def fetch_field(self, oid, field):
        raise NotImplementedError

    def save(self, obj):
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError

    def delete(self, oid):
        raise NotImplementedError

    def fetch_all(self):
        raise NotImplementedError

    def fetch_many(self, oids):
        raise NotImplementedError

    def inc(self, oid, field, value):
        raise NotImplementedError

    def inc_many(self, inc_op_iter):
        raise NotImplementedError

    def to_db_field(self, field):
        return field

    def from_db_field(self, field):
        return field

    def iter_ids(self):
        raise NotImplementedError

    # naive bulk get/set methods expected to be overridden
    def fetch_many(self, oid_iter):
        for oid in oid_iter:
            yield self.get(oid)

    def save_many(self, obj_iter):
        for obj in obj_iter:
            self.save(obj)

    def batched_inserter(self, batch_size):
        return BatchInserter(self, batch_size)

    @staticmethod
    def Get(store_id, **kwargs):
        uri = kwargs.pop('uri', None)
        uri = uri or os.environ.get(DATASTORE_URI_VAR, 'mongodb://localhost')
        proto = uri.split(':')[0] if uri else None

        store_cls = {
            'mongodb': MongoStore,
            'redis': RedisStore
        }.get(proto, None)

        if store_cls == None:
            log.error('Unsupported data store proto (%s), choose from (redis,mongodb)', proto)
            raise NotImplementedError

        return store_cls.Get(store_id, uri, **kwargs)

class RedisStore(Store):
    """ Abstract base class for stores built on redis """
    def __init__(self, namespace, uri):
        self.kvs = redis.from_url(uri)
        self.ns = namespace

    def to_key(self, oid):
        return self.ns + ':' + oid

    def to_oid(self, key):
        return key[len(self.ns)+1:].decode('utf-8')

    def _fetch_batch(self, keys_iter):
        raise NotImplementedError

    def fetch_many(self, oids):
        return self._fetch_batch(self.to_key(oid) for oid in oids)

    def fetch_all(self):
        keys = self.keys()
        keys_iter = islice(keys, None)
        if keys:
            batch_sz = 100000
            for _ in xrange(0, len(keys), batch_sz):
                for obj in self._fetch_batch(islice(keys_iter, batch_sz)):
                    yield obj

    def iter_ids(self):
        for key in self.keys():
            yield self.to_oid(key)

    def flush(self):
        self.kvs.eval("""
            local keys = redis.call('keys', '{:}*')
            for i=1,#keys,5000 do
                redis.call('del', unpack(keys, i, math.min(i+4999, #keys)))
            end
        """.format(re.escape(self.ns+':').replace('\\','\\\\')), 0)

    def delete(self, oid):
        self.kvs.delete(self.to_key(oid))

    def keys(self):
        return self.kvs.keys(re.escape(self.to_key('')) + '*')

    @classmethod
    def Get(cls, namespace, url='redis://localhost', flat = False):
        log.debug("Using %s redis data store for (%s)...", 'hashed' if flat else 'serialised', namespace)
        if flat:
            return HashedRedisStore(namespace, url)
        else:
            return SerialisedRedisStore(namespace, url, fmt = json)

class SerialisedRedisStore(RedisStore):
    def __init__(self, *args, **kwargs):
        self.fmt = kwargs.pop('fmt')
        super(SerialisedRedisStore, self).__init__(*args, **kwargs)

    def deserialise(self, data):
        return self.fmt.loads(data)

    def serialise(self, obj):
        return self.fmt.dumps(obj)

    def _fetch_batch(self, keys_iter):
        for data in self.kvs.mget(keys_iter):
            yield self.deserialise(data) 

    def fetch(self, oid):
        key = self.to_key(oid)
        data = self.kvs.get(key)
        return self.deserialise(data) if data != None else None

    def fetch_field(self, oid, field):
        # this is an inefficient operation for serialised redis stores
        return self.fetch(oid).get(field, None)

    def save(self, obj):
        key = self.to_key(obj['_id'])
        data = self.serialise(obj)
        self.kvs.set(key, data)

    def save_many(self, obj_iter):
        self.kvs.mset({
            self.to_key(obj['_id']): self.serialise(obj)
            for obj in obj_iter})

class HashedRedisStore(RedisStore):
    def __init__(self, *args, **kwargs):
        super(HashedRedisStore, self).__init__(*args, **kwargs)

    def fetch(self, oid):
        obj = self.kvs.hgetall(self.to_key(oid))
        if obj:
            obj['_id'] = oid
        else:
            obj = None
        return obj

    def fetch_field(self, oid, field):
        return self.kvs.hget(self.to_key(oid), field)

    def _fetch_batch(self, keys_iter):
        with self.kvs.pipeline(transaction=False) as p:
            keys = list(keys_iter)
            for key in keys:
                p.hgetall(key)
            for key, obj in izip(keys, p.execute()):
                obj['_id'] = self.to_oid(key)
                yield obj

    def save(self, obj):
        self._save(obj)
    
    def _save(self, obj, interface=None):
        interface = interface if interface != None else self.kvs
        key = self.to_key(obj['_id'])
        obj = dict(obj)
        obj.pop('_id', None)
        interface.hmset(key, obj)

    def save_many(self, obj_iter):
        with self.kvs.pipeline(transaction=False) as p:
            for obj in obj_iter:
                self._save(obj, interface=p)
            p.execute()

    def inc(self, oid, field, value):
        self._inc(oid, field, value)

    def _inc(self, oid, field, value, interface=None):
        interface = interface if interface != None else self.kvs
        interface.hincrby(self.to_key(oid), field, value)

    def inc_many(self, updates_by_oid_iter):
        with self.kvs.pipeline(transaction=False) as p:
            for oid, updates in updates_by_oid_iter:
                for field, value in updates:
                    self._inc(oid, field, value, interface=p)
            p.execute()

class MongoStore(Store):
    def __init__(self, db, collection, uri='mongodb://localhost'):
        self.collection = MongoClient(uri)[db][collection]

    def fetch(self, oid):
        return self.collection.find_one({'_id':oid})

    def fetch_field(self, oid, field):
        return self.collection.find_one({'_id':oid}, {field:True}).get(field, None)

    def fetch_all(self):
        return self.collection.find()

    def iter_ids(self):
        for obj in self.collection.find({}, {'_id':True}):
            yield obj['_id']

    def save(self, obj):
        self.collection.save(obj)        
   
    def save_many(self, obj_iter):
        self.collection.insert(obj_iter)

    def flush(self):
        self.collection.drop()

    def delete(self, oid):
        self.collection.delete_one({'_id':oid})

    def to_db_field(self, field):
        return field.replace('.', u'\u2024').replace('$',u'\uff04')

    def from_db_field(self, field):
        return field.replace(u'\u2024', '.').replace(u'\uff04','$')

    def inc(self, oid, field, value):
        self.inc_many([(oid,self.to_db_field(field),value)])

    def inc_many(self, updates_by_oid_iter):
        bulk = self.collection.initialize_unordered_bulk_op()
        for oid, updates in updates_by_oid_iter:
            if updates:
                bulk.find({
                    '_id': oid
                }).upsert().update_one({
                    '$inc': {self.to_db_field(f):v for f,v in updates.iteritems()}
                })
        bulk.execute()

    @classmethod
    def Get(cls, store_id, uri='mongodb://localhost', **kwargs):
        db, collection = store_id.split(':')

        log.debug("Using mongo data store (db=%s, collection=%s)...", db, collection)
        return MongoStore(db, collection, uri)

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
