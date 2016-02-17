from __future__ import absolute_import
import redis
import re
import ujson as json
from itertools import islice, izip

from ..data import Store, ObjectStore, FieldStore, SetStore

from nel import logging
log = logging.getLogger()

class RedisStore(Store):
    """ Abstract base class for stores built on redis """
    def __init__(self, namespace, uri):
        self.kvs = redis.from_url(uri)
        self.ns = namespace

    @classmethod
    def get_protocol(cls):
        return 'redis'

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

    def exists(self, oid):
        return self.kvs.exists(self.to_key(oid))

    def delete(self, oid):
        self.kvs.delete(self.to_key(oid))

    def keys(self):
        return self.kvs.keys(re.escape(self.to_key('')) + '*')

    @classmethod
    def Get(cls, store_id, uri='redis://localhost', **kwargs):
        return cls(store_id, uri, **kwargs)

class RedisObjectStore(RedisStore, ObjectStore):
    def __init__(self, *args, **kwargs):
        self.deserialise = kwargs.pop('deserializer', json.loads)
        self.serialise = kwargs.pop('serializer', json.dumps)
        super(RedisObjectStore, self).__init__(*args, **kwargs)

    def deserialise(self, data):
        return self.fmt.loads(data)

    def serialise(self, obj):
        return self.fmt.dumps(obj)

    def _fetch_batch(self, keys_iter):
        for data in self.kvs.mget(keys_iter):
            yield self.deserialise(data) if data != None else None 

    def fetch(self, oid):
        key = self.to_key(oid)
        data = self.kvs.get(key)
        return self.deserialise(data) if data != None else None

    def save(self, obj):
        key = self.to_key(obj['_id'])
        data = self.serialise(obj)
        self.kvs.set(key, data)

    def save_many(self, obj_iter):
        self.kvs.mset({
            self.to_key(obj['_id']): self.serialise(obj)
            for obj in obj_iter})

class RedisFieldStore(RedisStore, FieldStore):
    def __init__(self, *args, **kwargs):
        super(RedisFieldStore, self).__init__(*args, **kwargs)

    def fetch_fields(self, oid):
        return self.kvs.hgetall(self.to_key(oid))

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

    def set_fields(self, oid, fvs):
        self._set_fields(oid, fvs)

    def _set_fields(self, oid, fvs, interface=None):
        interface = interface if interface != None else self.kvs
        key = self.to_key(oid)
        interface.hmset(key, fvs)

    def set_fields_many(self, oid_fvs_iter):
        with self.kvs.pipeline(transaction=False) as p:
            for oid, fvs in oid_fvs_iter:
                self._save(oid, fvs, interface=p)
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
