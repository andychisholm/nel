# nel: The Entity Linking framework

__nel__ is an fast, accurate and highly modular framework for linking entities in documents.

Out of the box, __nel__ provides:

- named entity recognition
- coreference clustering and candidate generation
- multipple entity disambiguation feature models
- a supervised learning-to-rank framework for entity disambiguation
- a supervised nil detection system with configurable confidence thresholds
- basic nil clustering for out-of-KB entities
- support for evaluating linker performance and running error analysis

__nel__ is modular, it can:

- link entity mentions to any knowledge base you like (not just Wikipedia and Freebase!)
- update, rebuild and redeploy models as a knowledge base changes over time
- retrain recognition and disambiguation classifiers on your own corpus of documents
- adapt linking pipelines to meet performance, precision and recall tradeoffs

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
