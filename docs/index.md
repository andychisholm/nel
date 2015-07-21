# nel: A Python-based Entity Linking framework

__nel__ is an fast, accurate and highly modular framework for linking entities in documents.

Out of the box, __nel__ provides:

- support for efficient extraction and storage of entity information from sources like Wikipedia and Wikidata
- wrappers for state-of-the-art named entity recognition systems (Stanford, Schwa)
- full entity linking pipeline with coreference, candidate generation, supervised ranking, nil classification and clustering
- support for injestion and easy evaluation of linking systems over benchmark datasets (CoNLL, TAC)

## License

__nel__ is open-source software released under an [MIT license](http://opensource.org/licenses/MIT).

You're free to copy, modify and deploy the code in any setting you like - no strings attached.

## Getting started

__nel__ can be used in a variety of ways:

- via the command line
    - for offline, batch document linking
    - hosting an entity linking server
    - processing of evaluation corpora and training documents
    - constructing linking models over large datasets
- from python
    - for online document linking
    - model inspection and corpus analysis
