
# Overview

> Work in Progress

Document describing MongoDB based data models used by NEL.

## Online Components

These components are used in the online linking pipeline.

### Candidates

We need a way of mapping names we find in text to a set of candidate entities for that name.

#### Storage

Database  | Collection
--------- | ----------
models    | aliases

#### Schema

```
{
	'_id': string,
	'entities': list(string)
}
```

Field       | Content | Example
----------- | ------- | -------
_id | An entity alias | john smith
entities | The set of candidate entity ids for this alias | `['John_Smith', 'John_Smith_(actor)']`

> **Note:** Aliases are case normalised for case-insensitive candidate generation.

#### Queries
* Select by `_id`

### Classifiers

#### Storage

Database | Collection
-------- | ----------
models   | classifiers

#### Schema

```
{
	'_id': string,
	'weights': list(float),
	'corpus': string,
	'tag': string,
	'mapping': {
		'name': string,
		'params': {
			'features': list(string),
			'means': list(float),
			'stds': list(float)
		}
	}
}
```

Field | Content | Example
----- | ------- | -------
_id | Identifier string for the classifier. | default
weights | Feature weight vector | `[0.1, 0.2, 0.3]`
corpus | The corpus id this classifier was trained over | conll
tag | The corpus subset tag this classifier was trained over | train
mapping | Feature mapper configuration | 
mapping.name | Feature mapper class name | `PolynomialMapper`
mapping.params | Feature mapper class parameters | 
mapping.params.features | Ordered list of features | `['EntityProbability','NameProbability']`
mapping.params.features | Projected feature means | `[0.1, 0.2]`
mapping.params.features | Projected feature standard deviations | `[0.1, 0.2]` 

#### Queries

* Select by `_id`

## Offline Components

### Documents

Given a corpus of documents (e.g. CoNLL, TACs), we process documents from their source format into a common object model used throughout the linking pipeline.

Feature extraction, model training and linking operations may then be ran over each collection or some subset.

#### Storage

Database | Collection
-------- | ----------
docs     | `corpus_name`

> Where `corpus_name` is the name of the corpus, e.g. `conll`.

> **Note: ** Thinking of changing this to allow for easier cross-corpus queries.

#### Schema

```
{
	'_id': string,
	'tag': string,
	'text': string,
	'chains': [{
		'resolution': candidate,
		'candidates': [{
			'id': string,
			'features': {
				...
			}
		}]
		'mentions': [{
			'begin': int,
			'end': int,
			'text': int
		}]
	}]
}
```

Field       | Content | Example
----------- | ------- | -------
_id | A document identifier | Sport 123 testb
tag | A tag for the document within the corpus. Useful for tracking train/dev/test splits | train
text | Plaintext document content | The cat sat on the mat
chains | Clusters of coreferential mentions |
chains[`i`].resolution | The resolved entity for this chain |
chains[`i`].candidates | The set of candidates for this chain
chains[`i`].candidates[`j`].id | Entity id of the candidate | `John_Smith`
chains[`i`].candidates[`j`]features | Dictionary of feature name &rarr; value | `{'EntityProbability': 0.2}`
chains[`i`].mentions | A list of mentions in the chain |
chains[`i`].mentions[`j`].begin | Offset of this mention in the document text | 4
chains[`i`].mentions[`j`].end | End offset of this mention in the document text | 7
chains[`i`].mentions[`j`].text | Raw text of this mention, i.e. doc.text[begin:end] | cat

#### Queries

* Select by `tag`
