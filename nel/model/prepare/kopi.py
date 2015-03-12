import logging
from .util.kopireader import KopiReader

log = logging.getLogger()

class BuildKopiWikiTermCounts(object):
    "Builds entity term count model from KopiWiki counts data."
    def __init__(self, inpath, outpath):
        self.in_path = inpath
        self.out_path = outpath

    @staticmethod
    def iter_entity_term_counts(path):
        for line in iter_lines_from_files(path):
            columns = line.split(' ')
            yield columns[0], (c.split('|') for c in columns[1:])

    def __call__(self):
        self.build(self.in_path)#.write(self.out_path)

    def build(self, path):
        model = EntityTermCounts()
        
        # todo: parameterise
        wv_path = '/n/schwa11/data0/linking/erd/full/models'
        wv = WordVectors.read(os.path.join(wv_path, 'googlenews300.wordvector.model'))
        wv_vocab = set(wv.vocab.iterkeys())
        wv = None

        log.debug('Reading raw kopi counts...')
       
        def iter_term_counts():
            with open(self.out_path, 'wb') as f:
                for i, (entity, term_counts) in enumerate(BuildKopiWikiTermCounts.iter_entity_term_counts(self.in_path)):
                    if i % 100000 == 0: log.debug('Read %d entities...' % i)
                    yield (entity, {t:int(c) for t,c in term_counts if t in wv_vocab})

        mmdict.write(self.out_path, iter_term_counts())

        return model

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('inpath', metavar='INFILE')
        p.add_argument('outpath', metavar='OUTFILE')
        p.set_defaults(cls=cls)
        return p

MIN_ALIAS_LEN = 2
class BuildKopiWikiEntityContext(object):
    """
    Builds entity mention context collection from kopiwiki data.
    Assumes name and link models have already been built.
    """
    PUNCTUATION = frozenset(['.', '?', '!'])

    def __init__(self, kopiroot, kopidate, kopilang, aliaspath, linkspath, contextsize, outpath):
        self.alias_model_path = aliaspath
        self.link_model_path = linkspath
        self.context_size = None if contextsize <= 0 else contextsize
        self.out_path = outpath

        self.kopi_reader = KopiReader(kopiroot, kopidate, kopilang)
        self.init()

    def init(self):
        # todo: alias model
        log.debug('Loading entity alias model: %s' % self.alias_model_path)
        with open(self.alias_model_path, 'rb') as f:
            self.entity_names = marshal.load(f)

        self.link = Links.read(self.link_model_path) # {entity: [out_link_targets]} dictionary
    
        from nltk.corpus import stopwords
        from nltk.stem.wordnet import WordNetLemmatizer
        self.commoners = frozenset(iter_common_lemma_names())
        self.stops = frozenset(stopwords.words('english'))
        self.lemma = WordNetLemmatizer().lemmatize

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('kopiroot', metavar='KOPIROOT', help='Kopiwiki root')
        p.add_argument('kopidate', metavar='KOPIDATE', help='E.g., 20131203')
        p.add_argument('kopilang', metavar='KOPILANG', help='E.g., en')
        p.add_argument('aliaspath', metavar='ALIAS_MODEL_PATH', help='Path to entity alias model.')
        p.add_argument('linkspath', metavar='LINK_MODEL_PATH', help='Path to links model.')
        p.add_argument('contextsize', metavar='CONTEXTSIZE', type=int)
        p.add_argument('outpath', metavar='OUTFILE', help='Output file')
        p.set_defaults(cls=cls)
        return p

    def __call__(self):
        "Build model and write to file"
        entity_counts = defaultdict(Counter)

        # todo: fix
        wv = WordVectors.read('/n/schwa11/data0/linking/erd/full/models/googlenews300.wordvector.model')
        set_vocab(set(wv.vocab.iterkeys()))
        wv = None

        partitions = 4
        pid = int(platform.node()[-2:])-8 # schwa08 -> schwa11
        
        log.debug('Running against partition %i/%i...' % (pid+1, partitions))

        def process_kopi_file(path):
            article_count = 0
            entity_counts = defaultdict(Counter)

            for source, text in KopiReader.read(path):
                article_count += 1
                for entity, left, middle, right in self.contexts(source, text):
                    emw = (entity, [(left, middle, right)], self.context_size)

                    if random.randint(0, 999) == 0:
                        with open(self.out_path + '.sample', 'a') as f:
                            mention_text = ' ### '.join([left, middle, right]).replace('\t', '  ').replace('\n', ' ')
                            f.write('\t'.join([source, entity, middle, mention_text]).encode('utf-8') + '\n')

                    _, counts = get_mention_term_counts(emw)

                    # surprisingly this is much, much faster than
                    # the equivalent: entity_counts[entity] += counts
                    entity_counter = entity_counts[entity]
                    for k, v in counts.iteritems():
                        entity_counter[k] += v

            return (article_count, entity_counts)

        cumulative_article_count = 0
        cumulative_entity_counts = defaultdict(Counter)

        with parmapper(process_kopi_file, nprocs=1) as p:
            for i, (_, (articles_processed, entity_counts)) in enumerate(p.consume(self.kopi_reader.iter_files(partitions, pid))):
                cumulative_article_count += articles_processed
                log.debug('Processed %i kopi files. Accumulating counts...' % (i+1))

                for entity, counts in entity_counts.iteritems():
                    entity_counter = cumulative_entity_counts[entity]
                    for k, v in counts.iteritems():
                        entity_counter[k] += v

                log.debug('Processed %i articles...' % cumulative_article_count)

        log.debug('Writing mention context model (%i entities): %s' % (len(cumulative_entity_counts), self.out_path))
        mmdict.write(self.out_path + '.' + str(pid), ((k, dict(v)) for k,v in cumulative_entity_counts.iteritems()))

    def contexts(self, source, text):
        "Yield (entity, left_context, match, right_context) tuples"
        names = self.names(source)
        if names:
            name_pattern = '|'.join([re.escape(n) for n in names.keys()])
            for match in re.finditer(name_pattern, text):
                if self.is_common(match, text):
                    continue
                entity = names[match.group()]
                left, middle, right = self._contexts(match, text)
                yield entity, left, middle, right

    def _contexts(self, match, text):
        "Return (left_context, match, right_context) tuple)"
        # context_size is in tokens but we don't want to do tokenisation here
        # instead, we just want to return enough characters from the text
        # such that the tokeniser has enough to generate context_size tokens

        # avg token len = 5.1 chars + space character + 1
        context_chars = int(math.ceil((self.context_size / 2.0) * 7))

        start = max(match.start()-context_chars, 0)
        end = min(match.end()+context_chars, len(text))
        left = text[start:match.start()]
        middle = text[match.start():match.end()]
        right = text[match.end():end]
        return left, middle, right

    def is_common(self, match, text):
        "True if match might be a common word"
        if self.lemma(match.group()) in self.commoners:
            left_text = text[0:match.start()].strip()
            if left_text == '':
                # common word at text start
                return True
            left_char = left_text[-1]
            if left_char in self.PUNCTUATION:
                # common word at sentence start
                return True
            left_token = left_text.split()[-1]
            if not left_token.islower():
                # common word in possible title sequence
                return True
        return False

    def names(self, source):
        "Return {name: entity} dictionary"
        d = {}
        seen = set()
        for entity in self.link.get(source):
            for n in self.entity_names.get(entity, []):
                if n in seen:
                    # other entities mentioned in this text have same name
                    if n in d:
                        del d[n]
                elif n.islower(): pass              # exclude lowercase
                elif n in self.stops: pass          # name is a very common word
                elif len(n) < MIN_ALIAS_LEN: pass   # short names are likely to be ambiguous
                else:
                    d[n] = entity
                    seen.add(n)
        return d
