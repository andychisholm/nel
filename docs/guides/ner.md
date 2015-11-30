## Stanford NER

```
# install stanford NER tools
sudo yum -y install java wget unzip
wget http://nlp.stanford.edu/software/stanford-ner-2014-08-27.zip
unzip stanford-ner-2014-08-27.zip -d ner
rm stanford-ner-2014-08-27.zip
```

## libschwa NER

See: https://github.com/schwa-lab/libschwa/wiki/Installing

**Note:** On CentOS/AWS, you may need to update your `PKG_CONFIG_PATH` and `LD_LIBRARY_PATH` environment variables.

E.g., adding the following to `/etc/profile`:
```
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```
