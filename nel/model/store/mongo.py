from pymongo import MongoClient

from ..data import Store, ObjectStore, FieldStore, SetStore

from nel import logging
log = logging.getLogger()

class MongoStore(Store):
    def __init__(self, db, collection, uri='mongodb://localhost'):
        self.collection = MongoClient(uri)[db][collection]

    @classmethod
    def get_protocol(cls):
        return 'mongodb'

    def iter_ids(self):
        for obj in self.collection.find({}, {'_id':True}):
            yield obj['_id']

    def flush(self):
        self.collection.drop()

    def exists(self, oid):
        return self.collection.find({'_id':oid}).count() == 1

    def delete(self, oid):
        self.collection.delete_one({'_id':oid})

    def fetch(self, oid):
        return self.collection.find_one({'_id':oid})

    def fetch_all(self):
        return self.collection.find()

    def save(self, obj):
        self.collection.save(obj)        
   
    def save_many(self, obj_iter):
        self.collection.insert(obj_iter)

    @classmethod
    def Get(cls, store_id, uri='mongodb://localhost', **kwargs):
        db, collection = store_id.split(':')
        return cls(db, collection, uri)

class MongoObjectStore(MongoStore, ObjectStore):
    pass

class MongoFieldStore(MongoStore, FieldStore):
    @staticmethod
    def to_db_field(field):
        return field.replace('.', u'\u2024').replace('$',u'\uff04')

    @staticmethod
    def from_db_field(field):
        return field.replace(u'\u2024', '.').replace(u'\uff04','$')

    def fetch_fields(self, oid):
        obj = self.fetch(oid)
        if obj != None:
            obj.pop('_id')
            return {self.from_db_field(f):v for f,v in obj.iteritems()}
        return None

    def fetch_field(self, oid, field):
        field = self.to_db_field(field)
        return self.collection.find_one({'_id':oid}, {field:True}).get(field, None)

    def inc(self, oid, field, value):
        self.inc_many([(oid,self.to_db_field(field),value)])

    def inc_many(self, updates_by_oid_iter):
        bulk = self.collection.initialize_unordered_bulk_op()

        for oid, updates in updates_by_oid_iter:
            if updates:
                bulk.find({
                    '_id': oid
                }).upsert().update_one({
                    '$inc': {self.to_db_field(f):v for f,v in updates}
                })

        bulk.execute()
