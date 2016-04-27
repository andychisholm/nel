# Building Wikipedia Models

This example demonstrates the process of downloading, preprocessing and building models of entities in Wikipedia.

## Download and Preprocess the latest Wikipedia Dump

__sift__ provides a handy script for downloading the latest partitioned Wikipedia dump.

```
download-wikipedia latest
```

Once the download completes, the raw, compressed media-wiki partitions of Wikipedia will be under the `latest` directory.

First, we must extract redirect mappings to properly resolve inter-article links from this version of the dump.

```
sift build-corpus --save redirects WikipediaRedirects latest json
```

The extracted redirect mappings are now stored under the `redirects` directory.

Next, we perform full plain-text extraction over the Wikipedia dump, mapping links to their current Wikipedia target.

```
sift build-corpus --save processed WikipediaArticles latest --redirects redirects json
```

Our wikipedia corpus is now in the standard format for __sift__ corpora from which models can be extracted.

## Build Feature Models

We will now extract two simple count driven models from this corpus which are useful in entity linking.

The first model, "EntityCounts" simply collects the total count of inlinks for each entity over the corpus.

We use this statistic as a proxy for the prior probability of an entity and expect that entities with higher counts are more likely to be linked.

```
sift build-doc-model --save ecounts EntityCounts processed redis --prefix models:ecounts[wikipedia]:
```

The second model, "EntityNameCounts" collects the number of times a given anchor text string is used to link an entity.

This statistic helps us model the conditional probability of an entity given the name used to reference it in text.

```
sift build-doc-model --save necounts EntityNameCounts processed --lowercase redis --prefix models:necounts[wikipedia]:
```

## Push Models into the Data Store

To access feature models efficiently at runtime, __nel__ requires that feature models are stored in either redis or mongodb.

When building models with __sift__, you must specify an appropriate output format for use with your chosen __nel__ data store.

### Redis

In the example above, we selected the 'redis' output format for models.

This generates redis protocol which can be piped directly into a local redis instance via redis-cli:

```
zcat -r ecounts/*.gz | redis-cli --pipe
zcat -r necounts/*.gz | redis-cli --pipe
```

### Mongo

Alternatively, if the 'json' output format is selected, we can import into mongodb using mongoimport:

```
zcat -r ecounts/*.gz | mongoimport --db models --collection ecounts[wikipedia]
zcat -r necounts/*.gz | mongoimport --db models --collection necounts[wikipedia]
```

## Inspect the Models

```python
from nel.model.disambiguation import EntityCounts

ec = EntityCounts('wikipedia')
ec.count('en.wikipedia.org/wiki/Apple_Inc.')
```
