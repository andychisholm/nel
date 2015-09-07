# Overview

__nel__ builds linking models over collections of documents in the [wikijson](https://github.com/wikilinks/wikijson) format.

# Prerequisites
```
pip install git+http://git@github.com/wikilinks/wikijson.git
```

# Example
```
# download the latest wikipedia dump
wget http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 -O wikipedia.xml.bz2

# preprocess the Wikipedia dump with wikijson
wikijson process-dump wikipedia.xml.bz2 wikipedia.js.gz

# extract redirects mappings
nel build-wikipedia-redirects wikipedia.xml.bz2

# build linking models over inlinks and comentions
nel build-link-models wikipedia.js.gz wikipedia
```
