import json
from HTMLParser import HTMLParser
from re import findall, match, DOTALL, sub, compile
from collections import defaultdict
from nel.doc import Doc

def normalize_special_characters(text):
    """Replace special unicode characters in text"""
    entities = {
        u'\u00A0': ' ',
        u'&quot;': '"     ',
        u'\u0022': '"',
        u'\u00AB': '"',
        u'\u00BB': '"',
        u'\u02BA': '"',
        u'\u02DD': '"',
        u'\u030B': '"',
        u'\u030E': '"',
        u'\u05F4': '"',
        u'\u201C': '"',
        u'\u201D': '"',
        u'\u2036': '"',
        u'\u2033': '"',
        u'\u275D': '"',
        u'\u275E': '"',
        u'\u3003': '"',
        u'\u301D': '"',
        u'\u301E': '"',
        u'\U0001F676': '"',
        u'\U0001F677': '"',
    }
    for c, r in entities.iteritems():
        if c in text:
            text = text.replace(c, r)
    return text

def markdown_to_whitespace(markdown_txt):
    """Convert a text from markdown whitespace padded unicode."""
    resu = u''
    RE_MOVE_DOT = compile(r"(\s)([^\s]+)\\(\.\w)")
    RE_MOVE_DASH = compile(r"(\s)([^\s]+)\\(-\w)")

    # remove table
    match_table = findall(ur'(<table>.+?</table>)', markdown_txt, DOTALL)
    for match_t in match_table:
        markdown_txt = markdown_txt.replace(match_t, u' ' * len(match_t))
        
    for i, line in enumerate(markdown_txt.split('\n')):
        if i != 0:
            resu += u'\n'

        if len(line) > 0:
            # remove blockquote formatting
            if line.startswith(u'> '):
                line = u'  ' + line[2:]

            # remove headline formatting
            match_head = match(ur'^(#+) (.+)$', line)
            if match_head:
                sz = len(match_head.groups()[0])
                if sz < 7:
                    line = (' ' * sz) + line[sz:]

            # remove emphasized formatting
            emphs = findall(ur'((?<!\\)\*(.+?)(?<!\\)\*)', line)
            for emph in emphs:
                line = line.replace(emph[0], emph[0].replace('*', ' '))

            # remove important formatting
            imps = findall(ur'((?<!\\)__(.+?)(?<!\\)__)', line)
            for imp in imps:
                line = line.replace(imp[0], imp[0].replace('_', ' '))

            # remove link formatting
            links_txt = u''
            links = findall(ur'((?<!\\)\[(.+?)(?<!\\)\](?<!\\)(\(\S+(?<!\\)\)))', line)
            for link in links:
                line = line.replace(link[0], ' ' + link[1] + ' ' + (' ' * len(link[2])))
                links_txt = links_txt + link[1]

            # U\.S\. -> U.S., instead of U. S.
            line = sub(RE_MOVE_DOT, r"\1 \2\3", line)
            line = sub(RE_MOVE_DASH, r"\1 \2\3", line)

            left_replace = u"\\`*_{[(#+-!"
            right_replace = u"}])."

            # remove extra backslash
            for char in left_replace:
                line = line.replace(u"\\" + char, " " + char)
            for char in right_replace:
                line = line.replace(u"\\" + char, char + " ")

            # add the text if it contains more than a link
            resu += line

    # remove special unicode characters
    return normalize_special_characters(resu)

class MarkupStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self._pd_char = ' '
        self._pd = []
        self._pd_len = 0
        self._line_offsets = [0]

    def feed_data(self, data):
        offset = len(self.rawdata)

        while True:
            offset = data.find('\n', offset)
            if offset != -1:
                offset += 1
                self._line_offsets.append(offset)
            else:
                break

        self.feed(data)

    def get_raw_offset(self):
        return self._line_offsets[self.lineno-1] + self.offset

    def append_data(self, data):
        self._pd.append(data)
        self._pd_len += len(data)

    def append_padding(self):
        offset = self.get_raw_offset()
        if self._pd_len != offset:
            self.append_data(self._pd_char * (offset - self._pd_len))

    def handle_entityref(self, name):
        self.append_data('&'+name+';')

    def handle_charref(self, ref):
        self.append_data('&#'+ref+';')

    def handle_data(self, data):
        self.append_padding()
        self.append_data(data)

    def handle_endtag(self, tag):
        self.append_padding()

    def get_padded_data(self):
        self.append_padding()
        return ''.join(self._pd)

def markup_to_whitespace(ml):
    parser = MarkupStripper()
    parser.feed_data(ml)
    return parser.get_padded_data()

def mention_to_neleval(doc, chain, mention):
    default_probability = 1.0
    default_type = 'UNK'

    result = u'\t'.join([
        doc.id,
        str(mention.begin),
        str(mention.end),
        chain.resolution.id if chain.resolution else 'NIL',
        str(default_probability),
        mention.tag if mention.tag else 'UNK'
    ])

    return result

def inject_html_links(raw, linked_doc, kb_prefix=None):
    return inject_links(raw, linked_doc, u'<a href="{kb_pfx}{entity}">{mention}</a>', kb_prefix=kb_prefix)

def inject_markdown_links(raw, linked_doc, kb_prefix=None):
    return inject_links(raw, linked_doc, u'[{mention}]({kb_pfx}{entity})', kb_prefix=kb_prefix)

def inject_links(raw, linked_doc, link_format, kb_prefix=None):
    kb_prefix = 'https://en.wikipedia.org/wiki/' if kb_prefix == None else kb_prefix

    mentions_by_offset = sorted(
        (m.begin, m, c.resolution.id)
        for c in linked_doc.chains
            for m in c.mentions
                if c.resolution != None)

    output = []
    last_offset = None
    for offset, mention, entity_id in mentions_by_offset:
        output.append(raw[last_offset:offset])

        # todo: handle escaping
        output.append(
            link_format.format(
                mention = mention.text,
                kb_pfx = kb_prefix,
                entity = entity_id))

        last_offset = mention.end

    output.append(raw[last_offset:None])
    return ''.join(output)

def to_neleval(doc):
    return u'\n'.join(mention_to_neleval(doc, chain, m) for chain in doc.chains for m in chain.mentions)

def to_json(doc):
    return json.dumps({
        'id':doc.id,
        'mentions':[{
            'begin': m.begin,
            'length': len(m.text),
            'text': m.text,
            'resolution': None if chain.resolution == None else {
                'entity': chain.resolution.id,
                'features': chain.resolution.features
            }
        } for chain in doc.chains for m in chain.mentions]})

def from_sift(doc):
    links_by_target = defaultdict(list)
    for m in doc['links']:
        links_by_target[m['target']].append(m)

    return Doc.obj({
        'id': doc['_id'],
        'text': doc['text'],
        'tag': None,
        'chains': [{
            'resolution': {'id': target},
            'mentions': [{
                'resolution': {'id': target},
                'begin': l['start'],
                'text': doc['text'][l['start']:l['stop']],
            } for l in links]
        } for target, links in links_by_target.iteritems()]
    })

def to_sift(doc):
    return json.dumps({
        '_id': doc.id,
        'text': doc.text,
        'links': [{
            'target': m.resolution.id,
            'start': m.span.start,
            'stop': m.span.stop
        } for chain in doc.chains for m in chain.mentions]
    })
