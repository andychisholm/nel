#!/usr/bin/env python
class Doc(object):
    def __init__(self, text, doc_id=None, tag=None, chains=None, _id=None, raw=None):
        self.text = text
        self.id = doc_id
        self._id = _id
        self.tag = tag
        self.chains = chains or []
        self.raw = raw

    def __str__(self):
        return '[Doc(%s)]' % self.id

    def json(self):
        jd = {
            'id': self.id,
            'tag': self.tag,
            'text': self.text,
            'raw': self.raw,
            'chains': [c.json() for c in self.chains]
        }
        if self._id != None:
            jd['_id'] = self._id
        return jd

    @staticmethod
    def obj(json):
        return Doc(
            json['text'],
            doc_id=json['id'],
            tag=json['tag'],
            chains=[Chain.obj(c) for c in json['chains']],
            _id=json.get('_id',None),
            raw=json.get('raw',None))

class Chain(object):
    """ Chain of coreferential mentions. """
    def __init__(self, mentions=None, candidates=None, resolution=None):
        self.mentions = mentions or []
        self.candidates = candidates or []
        self.resolution = resolution

    def json(self):
        return {
            'mentions': [m.json() for m in self.mentions],
            'candidates': [c.json() for c in self.candidates],
            'resolution': self.resolution.json() if self.resolution else None
        }

    @staticmethod
    def obj(json):
        return Chain(
            [Mention.obj(m) for m in json['mentions']],
            [Candidate.obj(c) for c in json.get('candidates', [])],
            Candidate.obj(json['resolution']) if json['resolution'] else None)

class Mention(object):
    def __init__(self, begin, text, tag=None, mid=None, resolution=None):
        self.begin = begin
        self.text = text
        self.tag = tag
        self.mid = mid
        self.resolution = resolution

    def json(self):
        return {
            'begin': self.begin,
            'end': self.end,
            'text': self.text,
            'tag': self.tag,
            'mid': self.mid,
            'resolution': self.resolution.json() if self.resolution else None
        }

    @property
    def span(self):
        return slice(self.begin, self.end)

    @staticmethod
    def obj(json):
        return Mention(
            json['begin'],
            json['text'],
            json.get('tag', None),
            json.get('mid', None),
            Candidate.obj(json['resolution']) if 'resolution' in json else None)

    @property
    def end(self):
        return self.begin + len(self.text)

    def __len__(self):
        return len(self.text)

class Candidate(object):
    def __init__(self, entity_id, features=None, fv=None):
        self.id = entity_id
        self.features = features or {}
        self.fv = fv

    def json(self):
        return {
            'id': self.id,
            'features': self.features
        }

    @staticmethod
    def obj(json):
        return Candidate(json['id'], json.get('features'))

