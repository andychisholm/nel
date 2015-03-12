import urllib
import json
import logging

log = logging.getLogger()

class BuildFreebaseCandidates(object):
    "Build candidate set from freebase api."
    def __init__(self):
        self.api_key = 'AIzaSyCVY0pW6R29lTdbLRmGHbZxgI8FFKl0DlE'

    def __call__(self):
        api_key = self.api_key

        query = 'Polish'

        log.debug('Searching freebase for (%s)' % query)
        service_url = 'https://www.googleapis.com/freebase/v1/search'
        params = {
                'key': api_key,
                'query': query,
        }
        url = service_url + '?' + urllib.urlencode(params)
        response = json.loads(urllib.urlopen(url).read()) 
        freebase_candidates = [(r['mid'], r['score']) for r in response['result']]

        #log.debug(freebase_candidates)
        log.debug('Querying freebase for page ids...')
        query = [{
            "mid|=": [mid for mid, _ in freebase_candidates if mid[:3] == '/m/'],
            "wiki_en:key": [{
                "/type/key/namespace": "/wikipedia/en_id",
                "value": None
            }]
        }]
        service_url = 'https://www.googleapis.com/freebase/v1/mqlread'
        #log.debug(query)
        params = {
                'key': api_key,
                'query': json.dumps(query),
        }
        url = service_url + '?' + urllib.urlencode(params)

        response = json.loads(urllib.urlopen(url).read())

        candidates = []
        for i, results in enumerate(response['result']):
            mid, score = freebase_candidates[i]
            for result in results["wiki_en:key"]:
                candidates.append((mid, score, result["value"]))

        service_url = 'http://en.wikipedia.org/w/api.php'
        params = {
            'action': 'query',
            'prop': 'info',
            'pageids': '|'.join(pid for _, _, pid in candidates),
            'inprop': 'url',
            'format': 'json'
        }
        url = service_url + '?' + urllib.urlencode(params)
        log.debug('Querying wikipedia for page titles...')
        response = urllib.urlopen(url).read()
        response = json.loads(response)
        #log.debug(response)
        for mid, score, pid in candidates:
            if 'fullurl' in response['query']['pages'][pid]:
                wk_url = response['query']['pages'][pid]['fullurl']
                wk_id = urllib.unquote(wk_url[wk_url.rfind('/')+1:])
                log.debug("%s\t%s\t%s\t%s" % (score, mid, pid, wk_id))
    
    @classmethod
    def add_arguments(cls, p):
        p.set_defaults(cls=cls)
        return p