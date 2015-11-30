# Installation

## Prerequisites

```
# system packages
sudo yum -y install gcc gcc-c++ gcc-gfortran gcc44-gfortran libgfortran lapack blas python-devel blas-devel lapack-devel

# install git
sudo yum -y install git

# install pip
sudo yum -y install epel-release
sudo yum -y install python-pip

# install virtual env
sudo pip install virtualenv
```

## Install nel

```
virtualenv ve
. ve/bin/activate

pip install git+http://git@github.com/wikilinks/nel.git
```

## Install a NER

The easiest way to get started is to install the spaCy NER system and models.

```
pip install spaCy
python -m spacy.en.download all
```

Alternatively, checkout the [NER guide](guides/ner.md) for other options.

## Setup the Model Store

To store linking models and work with offline corpora, nel requires access to some kind of data store.

Currently, redis and mongodb are supported.

To configure the datastore, you must set the `NEL_DATASTORE_URI` environment variable.

Redis should be preferred whenever linking models fit in memory as it allows for very fast model lookups at runtime.

For example, a local redis instance may be configured as:
```
export NEL_DATASTORE_URI='redis://localhost'
```
