import logging

log = logging.getLogger()

class BuildYagoMeansNames(object):
    "Builds entity alias model and entity title set from yago-means data."
    def __init__(self, inpath, redirect_model_path, alias_model_outpath, title_model_outpath):
        self.in_path = inpath
        self.alias_model_path = alias_model_outpath
        self.title_model_path = title_model_outpath

        log.info('Loading redirect model: %s ...', redirect_model_path)
        self.redirect_model = marshal.load(open(redirect_model_path, 'rb'))

    def __call__(self):
        log.debug('Loading entity alias model from Yago means dataset: %s ...' % self.in_path)

        d = {}
        name_count = 0
        for name, entity in iter_tsv(self.in_path, 2):
            name = name.decode('unicode-escape')
            entity = entity.decode('unicode-escape')
            entity = self.redirect_model.get(entity, entity)

            if entity not in d:
                d[entity] = []

            d[entity].append(name)
            name_count += 1

        log.debug('Writing entity alias model (%i entities, %i aliases) to file: %s ...' % (len(d), name_count, self.alias_model_path))
        with open(self.alias_model_path, 'wb') as f:
            marshal.dump(d, f)

        log.debug('Writing entity titles (%i entities): %s' % (len(d), self.title_model_path))
        with open(self.title_model_path, 'wb') as f:
            marshal.dump(d.keys(), f)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('inpath', metavar='INFILE')
        p.add_argument('redirect_model_path', metavar='REDIRECT_MODEL_PATH')
        p.add_argument('alias_model_outpath', metavar='OUT_ALIAS_MODEL_FILE')
        p.add_argument('title_model_outpath', metavar='OUT_TITLE_MODEL_FILE')
        p.set_defaults(cls=cls)
        return p