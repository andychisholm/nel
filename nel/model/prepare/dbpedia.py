from ..model import Entity, Name

from .util.nqstream import NqStream
from .util.ntstream import NtStream

import logging

log = logging.getLogger()

class BuildDbpediaLexicalisations(object):
    "Build entity and name models from DBpedia Spotlight lexicalisations file."
    ENTITY_SCORE = u'http://dbpedia.org/spotlight/score#uriProbability'
    ENTITY_PRE = u'http://dbpedia.org/resource/'
    NAME_SCORE = u'http://dbpedia.org/spotlight/score#uriGivenSf'
    NAME_PRE = u'http://dbepdia.org/spotlight/id/'

    def __init__(self, inpath, redirect_model_path, name_outpath, entity_outpath):
        self.in_path = inpath
        self.redirect_model_path = redirect_model_path
        self.name_out_path = name_outpath
        self.entity_out_path = entity_outpath

        log.info('Loading redirect model: %s ...', self.redirect_model_path)
        self.redirect_model = marshal.load(open(self.redirect_model_path, 'rb'))

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('inpath', metavar='INFILE')
        p.add_argument('redirect_model_path', metavar='REDIRECT_MODEL_PATH')
        p.add_argument('name_outpath', metavar='NAME_MODEL_OUTFILE')
        p.add_argument('entity_outpath', metavar='ENTITY_MODEL_OUTFILE')
        p.set_defaults(cls=cls)
        return p

    def __call__(self):
        entity = Entity()
        names = Name()

        entity_count = 0

        with open(self.in_path, 'rb') as fh:
            stream = NqStream()
            for subject, predicate, obj, _ in stream.iter(fh):
                if predicate.toPython() == self.ENTITY_SCORE:
                    #print 'Entity subject: %s' % subject.toPython().encode('utf8')
                    title = self._to_title(subject.toPython())
                    score = obj.toPython()
                    entity.update(title, score)

                    entity_count += 1
                    if entity_count % 100000 == 0:
                        log.debug('Processed %i entities...' % entity_count)

                elif predicate.toPython() == self.NAME_SCORE:
                    #print 'Name subject: %s' % subject.toPython().encode('utf8')
                    title, name = self._to_name(subject.toPython())
                    score = obj.toPython()

                    if title  == 'Pisco' or title == 'Carnoustie' or title == 'Bengkulu':
                        log.debug('%s|#|%s|#|%s', score, title, name)
                    names.update(name, title, score)

        entity.write(self.entity_out_path)
        names.write(self.name_out_path)

    def _trim(self, uri, prefix):
        assert uri.startswith(prefix), 'not valid subject'
        return unquote(uri.encode('utf8')[len(prefix):]).decode('utf8')

    def _to_title(self, uri):
        "Return final component of uri path."
        title = self._trim(uri, self.ENTITY_PRE)
        return self.redirect_model.get(title, title)

    def _normalise(self, name):
        "Return normalised name."
        return ' '.join(name.split('_'))

    def _to_name(self, uri):
        "Return split and normalised title and name."
        title, name = self._trim(uri, self.NAME_PRE).split('---', 1)
        return self.redirect_model.get(title, title), self._normalise(name)

class BuildDbpediaLinks(object):
    "Build cooccurrence and outlink models from DBpedia pagelinks nt."
    ENTITY_PRE = u'http://dbpedia.org/resource/'

    def __init__(self, fname, redirect_model_path, outpath):
        self.in_path = fname
        self.out_path = outpath
        self.namespace_re = re.compile(NAMESPACE_RE, re.I)
        self.pseudonamespace_re = re.compile(PSEUDONAMESPACE_RE)

        log.info('Loading redirect model: %s ...', redirect_model_path)
        self.redirect_model = marshal.load(open(redirect_model_path,'rb'))

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('fname', metavar='INFILE')
        p.add_argument('redirect_model_path', metavar='REDIRECT_MODEL_PATH')
        p.add_argument('outpath', metavar='OUTFILE')
        p.set_defaults(cls=cls)
        return p

    def __call__(self):
        self.build().write(self.out_path)

    def build(self):
        log.info('Building DBpedia links model from file: %s' % self.in_path)
        links = Links()
        for i, (source, target) in enumerate(self.read()):
            if i % 500000 == 0: log.debug('Processed %i links...' % i)

            source = self.redirect_model.get(source, source)
            target = self.redirect_model.get(target, target)

            links.update(source, target)
        return links

    def read(self):
        o = bz2.BZ2File if self.in_path.endswith('bz2') else open
        fh = o(self.in_path)
        stream = NtStream()
        for subject, predicate, obj in stream.iter(fh):
            source = self._to_title(subject.toPython())
            if not self.is_article(source):
                continue
            target = self._to_title(obj.toPython())
            if not self.is_article(target):
                continue
            yield source, target

    def _to_title(self, uri):
        "Return final component of uri path."
        return self._trim(uri, self.ENTITY_PRE)

    def _trim(self, uri, prefix):
        assert uri.startswith(prefix), 'not valid subject'
        return unquote(uri.encode('utf8')[len(prefix):]).decode('utf8')

    def is_article(self, title):
        "Return true if page is an article."
        if title == '':
            return False # empty name
        if title.endswith(DISAMBIGUATION_CLOSE):
            return False # disambiguation page
        if self.namespace_re.match(title):
            return False # not main article namespace
        if self.pseudonamespace_re.match(title):
            return False # not main article  namespace
        return True

class BuildDbpediaRedirects(object):
    "Build redirect model from DBpedia (transitive) redirects nt."
    ENTITY_PRE = u'http://dbpedia.org/resource/'

    def __init__(self, fname, outpath):
        self.in_path = fname
        self.out_path = outpath
        self.namespace_re = re.compile(NAMESPACE_RE, re.I)
        self.pseudonamespace_re = re.compile(PSEUDONAMESPACE_RE)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('fname', metavar='FILE')
        p.add_argument('outpath', metavar='OUTFILE')
        p.set_defaults(cls=cls)
        return p

    def __call__(self):
        redirects = Redirects()
        for source, target in self.read():
            # TODO check whether dbpedia titles need to be mapped to wikipedia
            # TODO applies to BuildLexicalisations as well
            redirects.update(source, target)
        
        redirects.write(self.out_path)

    @property
    def fh(self):
        if self.in_path.endswith('bz2'):
            #return bz2.BZ2File(self.in_path)
            # TODO workaround until bz2 multistream update
            return os.popen('bzip2 -cd {}'.format(self.in_path))
        else:
            return open(self.in_path, 'rb')

    def read(self):
        stream = NtStream()
        for subject, _, obj in stream.iter(self.fh):
            source = self._to_title(subject.toPython())
            target = self._to_title(obj.toPython())
            if not self.is_article(target):
                continue
            yield source, target

    def _to_title(self, uri):
        "Return final component of uri path."
        return self._trim(uri, self.ENTITY_PRE)

    def _trim(self, uri, prefix):
        assert uri.startswith(prefix), 'not valid subject'
        return unquote(uri.encode('utf8')[len(prefix):]).decode('utf8')

    def is_article(self, title):
        "Return true if page is an article."
        if title == '':
            return False # empty name
        if title.endswith(DISAMBIGUATION_CLOSE):
            return False # disambiguation page
        if self.namespace_re.match(title):
            return False # not main article namespace
        if self.pseudonamespace_re.match(title):
            return False # not main article  namespace
        return True
