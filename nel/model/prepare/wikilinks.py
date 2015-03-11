from time import time
from thrift.protocol import TBinaryProtocol
from thrift.transport import TTransport
from .util import trim_subsection_link
from .thrifty.wikilinks import ttypes as wikilinks

import logging
log = logging.getLogger()

class BuildWikilinksLexicalisations(object):
    "Builds lexicalisation models from an EntityMentionContext model."
    def __init__(self, inpath, outdir):
        self.in_path = inpath
        self.out_dir = outdir

    def __call__(self):
        log.debug('Generating name and entity prior models from wikilinks mention contexts...')
        
        mention_iter = EntityMentionContexts.iter_entity_mentions_from_path(self.in_path)
        
        entity_given_name_model = Name()
        #name_given_entity_model = Name()
        entity_model = Entity()

        for i, (entity, mentions) in enumerate(mention_iter):
            if i % 150000 == 0: log.debug('Processed %i entities...' % i)
            entity_model.update(entity, len(mentions))
            for _, _, name, _ in mentions:
                count = entity_given_name_model.d.get(name,{}).get(entity, 0.0)
                entity_given_name_model.update(name, entity, count + 1)

                #count = name_given_entity_model.score(entity, name)
                #name_given_entity_model.update(entity, name, count + 1)

        #log.debug('Saving raw name counts model...')
        #entity_given_name_model.write(os.path.join(self.out_dir,'wikilinks.name_count.model'))

        log.debug('Converting P(e|name) model counts to probabilities...')
        entity_given_name_model.normalise()
        entity_given_name_model.write(os.path.join(self.out_dir,'wikilinks.name.model'))

        #log.debug('Converting P(name|e) model counts to probabilities...')
        #name_given_entity_model.normalise()
        #name_given_entity_model.write(os.path.join(self.out_dir,'wikilinks.prob_n_e.model'))

        log.debug('Converting entity model counts to probabilities...')
        entity_model.normalise()
        entity_model.write(os.path.join(self.out_dir,'wikilinks.entity.model'))
    
    @classmethod
    def add_arguments(cls, p):
        p.add_argument('inpath', metavar='INFILE')
        p.add_argument('outdir', metavar='OUTDIR')
        p.set_defaults(cls=cls)
        return p

class BuildWikilinksEntityContext(object):
    "Builds entity mention context collection from wikilinks data."
    def __init__(self, inpath, outdir):
        self.in_path = inpath
        self.out_path = outdir

    def __call__(self):
        model_name = 'wikilinks.entity_context'
        
        # todo: fix wv_vocab at module scope
        wv = WordVectors.read('/n/schwa11/data0/linking/models/googlenews300.wordvector.model')
        set_vocab(set(wv.vocab.iterkeys()))
        wv = None

        low_mem_stream = True
        if not low_mem_stream:
            emcz = EntityMentionContexts.read(self.in_path)
        
        for window in [None]:
            window_model_name = 'w_' + str(window).lower()
            
            log.debug('Generating wikilinks entity context model for window size: %s ...' % str(window))

            count_model_fn = '%s.%s.tf.model' % (model_name, window_model_name)
            idf_model_fn = '%s.%s.idf.model' % (model_name, window_model_name)
            
            count_model_path = os.path.join(self.out_path, count_model_fn)
            idf_model_path = os.path.join(self.out_path, idf_model_fn)

            if low_mem_stream:
                iter_entity_mentions = EntityMentionContexts.iter_entity_mentions_from_path(self.in_path)
                iteration = ((e, mentions, window) for e, mentions in iter_entity_mentions)
            else:
                iteration = ((e, mentions, window) for e, mentions in emcz.entity_mentions.iteritems())

            def iter_filtered_counts(counts_iter, outputs):
                i = 0
                for _, (entity, counts) in counts_iter:
                    if i % 150000 == 0: log.debug('Processed %i entities...' % i)
                    
                    yield (entity, dict(counts))
                    i += 1

                    #log.debug(outputs)
                    # dirty computed df stats during iteration
                    outputs[0] += len(counts)
                    outputs[1].update(counts.iterkeys())

            log.debug('Streaming wikilink mentions into count model...')

            idf_stats = [0, Counter()]
            with parmapper(get_mention_term_counts, 30) as p:
                mmdict.write(count_model_path, iter_filtered_counts(p.consume(iteration), idf_stats))

            log.debug('Wrote count model: %s' % count_model_path)

            doc_count, term_dfs = idf_stats

            def iter_term_idfs():
                for t, df in term_dfs.iteritems():
                    yield (t, math.log(doc_count/df))

            log.debug('Writing idf model: %s' % idf_model_path)
            mmdict.write(idf_model_path, iter_term_idfs())

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('inpath', metavar='INFILE')
        p.add_argument('outdir', metavar='OUTDIR')
        p.set_defaults(cls=cls)
        return p

class BuildWikilinksMentions(object):
    "Builds entity mention context collection from wikilinks data."
    def __init__(self, wikilinksdir, redirect_model_path, outpath):
        self.wikilinks_dir = wikilinksdir
        self.out_path = outpath

        log.debug('Loading redirect map...')
        self.redirect_model = marshal.load(open(redirect_model_path, 'rb'))

    @staticmethod
    def iter_wikilink_items_under_path(path):
        for fopen in iter_fopens_at_path(os.path.join(path, '*.gz')):
            with fopen() as f:
                # there must be a nicer way to do this...
                thrift_buffer = TTransport.TFileObjectTransport(f)
                proto = TBinaryProtocol.TBinaryProtocol(thrift_buffer)
                while True:
                    try:
                        w = wikilinks.WikiLinkItem() 
                        wikilinks.WikiLinkItem.read(w, proto)
                        yield w
                    except EOFError: break

    @staticmethod
    def trim_protocol(s):
        idx = s.find('://')
        return s if idx == -1 else s[idx+3:]

    @staticmethod
    def unquote_and_decode(s):
        try:
            return urllib.unquote(s).decode('utf-8')
        except UnicodeDecodeError:
            return urllib.unquote(s.decode('utf-8'))

    def __call__(self):
        ems = EntityMentionContexts()

        total_mention_count = 0
        no_context_count = 0
        bad_entity_url_count = 0

        start_time = time()

        log.info('Processing wikilinks items...')
        for i, wli in enumerate(self.iter_wikilink_items_under_path(self.wikilinks_dir)):
            if i % 500000 == 0:
                log.debug('Processed %i pages... %.1f p/s', i, i / (time() - start_time))

            total_mention_count += len(wli.mentions)
            source = self.unquote_and_decode(wli.url)
            source = self.trim_protocol(self.trim_subsection_link(source))

            for m in wli.mentions:
                if m.context != None:
                    entity_url_section_idx = m.wiki_url.rfind('/')
                    if entity_url_section_idx != -1:
                        entity = m.wiki_url[entity_url_section_idx+1:]
                        entity = self.unquote_and_decode(entity)
                        entity = trim_subsection_link(entity)
                        entity = entity.replace(' ', '_')
                        entity = self.redirect_model.get(entity, entity)

                        left = m.context.left.decode('utf-8')
                        middle = m.context.middle.decode('utf-8')
                        right = m.context.right.decode('utf-8')

                        ems.add(entity, source, left, middle, right)
                    else: bad_entity_url_count += 1
                else: no_context_count += 1

        log.info(
            'Processed %i total mentions, %i with no context, %i with invalid uris' % 
            (total_mention_count, no_context_count, bad_entity_url_count))

        ems.write(self.out_path)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('wikilinksdir', metavar='WIKILINKS_DIR')
        p.add_argument('redirect_model_path', metavar='REDIRECT_MODEL_PATH')
        p.add_argument('outpath', metavar='OUTPATH')
        p.set_defaults(cls=cls)
        return p
