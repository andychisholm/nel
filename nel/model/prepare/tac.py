import HTMLParser

from ..model import Redirects

import logging
log = logging.getLogger()

class BuildTacRedirects(object):
    """ Creates a redirect model which maps TAC entity ids to wikipedia entity ids """
    def __init__(self, tac_kb_map_path, wiki_redirect_model_tag, tac_redirect_model_tag):
        self.tac_kb_map_path = tac_kb_map_path
        self.wiki_redirect_model_tag = wiki_redirect_model_tag
        self.tac_redirect_model_tag = tac_redirect_model_tag

    def iter_mappings(self):
        html_parser = HTMLParser.HTMLParser()
        wiki_redirects = Redirects(self.wiki_redirect_model_tag)

        log.debug("Reading tac entity kb: %s ...", self.tac_kb_map_path)
        with open(self.tac_kb_map_path, 'r') as f:
            for line in f:
                line = html_parser.unescape(line.decode('utf-8')).strip()
                tac_entity, wk_entity = line.split('\t')
                yield tac_entity, wiki_redirects.map(wk_entity)

    def __call__(self):
        log.info("Populating tac to wikipedia entity redirect mapping...")
        tac_redirects = Redirects(self.tac_redirect_model_tag)
        tac_redirects.create(self.iter_mappings())
        log.info("Done.")

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('tac_kb_map_path', metavar='TAC_KB_MAP_PATH')
        p.add_argument('--wiki_redirect_model_tag', default='wikipedia', required=False, metavar='WIKI_REDIRECTS')
        p.add_argument('--tac_redirect_model_tag', default='tac', required=False, metavar='MODEL_TAG')
        p.set_defaults(cls=cls)
        return p