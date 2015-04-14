Linker Setup from Scratch
===================

# Prerequisites

## Packages

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

# install stanford NER tools
sudo yum -y install java wget unzip
wget http://nlp.stanford.edu/software/stanford-ner-2014-08-27.zip
unzip stanford-ner-2014-08-27.zip -d ner
rm stanford-ner-2014-08-27.zip
```

## libschwa

See: https://github.com/schwa-lab/libschwa/wiki/Installing

**Note:** On CentOS/AWS, you may need to update your `PKG_CONFIG_PATH` and `LD_LIBRARY_PATH` environment variables.

E.g., adding the following to `/etc/profile`:
```
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```

# Installation

## Fetch Repo(s)

Fetch the linker **(Required)**:
```
git clone https://github.com/wikilinks/nel.git
```

Fetch the eval tool **(Optional)**:
```
git clone https://github.com/wikilinks/neleval.git
```

## Update Environment Paths

The `activate` script exports a bunch of environment variables then activates the python virtual environment.

* `NEL_ROOT` - path to the cloned [nel](https://github.com/wikilinks/nel) repository
* `NEL_MODELS` - path at which compiled linking models are stored
* `NEL_DATASTORE_URI` - URI which identifies the backing store for models (e.g. redis, mongo or file based)
* `NELEVAL_ROOT` - path to the cloned [neleval](https://github.com/wikilinks/neleval/) repository
* `STANFORD_NER_ROOT` - path to the extracted [Stanford NER](http://nlp.stanford.edu/software/CRF-NER.shtml) tools

Following on from above:
```
cd nel
mkdir -p data/models
vim activate
```

The `activate` script should look something like the following:
```
#!/bin/bash

export NEL_ROOT=~/nel
export NEL_MODELS_ROOT=$NEL_ROOT/data/models
export NEL_DATASTORE_URI='redis://localhost'

export NELEVAL_ROOT=~/neleval
export STANFORD_NER_ROOT=~/ner/stanford-ner-2014-08-27

. ve/bin/activate
```

## Setup the Virtual Environment

```
virtualenv ve
. activate
pip install -r requirements.txt
```

# Run Tests

Once they exist!

# Building Models

## Data sources

Wikipedia ([downloads](http://dumps.wikimedia.org/enwiki/), [latest](http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2))

Wikilinks ([downloads](http://www.iesl.cs.umass.edu/data/wiki-links#TOC-Dataset-with-Context))

# Running the Linker as a Service

## Run the NER Service
```
./scripts/ner_svc
```

## Prepare a Service Configuration File

Example: [nel/config/sample.config.json](https://github.com/wikilinks/nel/blob/master/config/sample.config.json)

## Run the Linking Service
```
./scripts/linker_svc foo/bar/config.json
```
