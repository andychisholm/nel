# nel: The Entity Linking framework

__nel__ is an fast, accurate and highly modular framework for linking entities in documents.

Out of the box, __nel__ provides:

- named entity recognition (DIY, or plug-in a NER system like Stanford, spaCy or Schwa)
- in-document coreference clustering
- candidate generation
- multiple disambiguation features
- a supervised learning-to-rank framework for entity disambiguation
- a supervised nil detection system with configurable confidence thresholds
- nil clustering
- support for evaluation and error analysis of linking system output

__nel__ is completely modular, it can:

- link entities to any knowledge base you like (not limited to just Wikipedia or Freebase)
- update, rebuild and redeploy linking models as a knowledge base changes over time
- retrain recognition and disambiguation models on your own corpus of documents
- easily adapt a linking pipeline to meet performance and accuracy tradeoffs

__nel__ is flexible, you can run it:

- ad-hoc, from python or as a web service
- offline, in parallel over a corpus of pre-processed documents
- with markdown, html or custom document formats (e.g. CoNLL, TAC)

## License

__nel__ is open-source software released under an [MIT license](http://opensource.org/licenses/MIT).

You're free to copy, modify and deploy the code in any setting you like - no strings attached.

## Getting started

### Installation

Checkout the [setup guide](installation.md) for details.

```
pip install git+http://git@github.com/wikilinks/nel.git
```

### Models

To link entities, __nel__ first needs some model of who or what an entity is.

__nel__ uses models from the [sift](https://github.com/wikilinks/sift) framework for entity linking.

To build models from scratch, you need a corpus of documents that link to entities in the KB.

Wikipedia is a good staring point for notable named entities.

See the [model build](guides/models.md) guide to get started.
