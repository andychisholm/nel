# Installation

For most environments, it should be as simple as:

```
pip install git+http://git@github.com/wikilinks/nel.git
```

## Setup a Model Store

To store linking models and work with offline corpora, nel requires access to some kind of data store.

Currently, redis and mongodb are supported.

To configure the datastore, you must set the `NEL_DATASTORE_URI` environment variable.

Redis should be preferred whenever linking models fit in memory as it allows for very fast model lookups at runtime.

For example, a local redis instance may be configured as:
```
export NEL_DATASTORE_URI='redis://localhost'
```

## Install a NER

The easiest way to get started is to install the spaCy NER system and models.

```
pip install spaCy
python -m spacy.en.download all
```

Alternatively, checkout the [NER guide](guides/ner.md) for other options.
