# Prerequisites

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

# Install the Linker

```
virtualenv ve
. ve/bin/activate

pip install git+http://git@github.com/wikilinks/nel.git
```

## Setting up a Data Store

To store linking models and work with offline corpora, nel requires access to some kind of data store.

Currently, redis and mongodb are supported.

To configure the datastore, you must set the `NEL_DATASTORE_URI` environment variable.

By default, mongodb is prefered:
```
export NEL_DATASTORE_URI='mongodb://localhost'
```

# Install the Eval Tools

```
pip install git+http://github.com/wikilinks/neleval.git#egg=neleval
```
