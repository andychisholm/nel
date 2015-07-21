# Download
```
WKDATA_DUMP_FN=20150420.json.gz
wget http://dumps.wikimedia.org/other/wikidata/$WKDATA_DUMP_FN
```

# Building Entity Set Models
```
nel build-wikidata-entity-set \
  $WKDATA_DUMP_FN \
    --include 5 \
    --include 2221906 \
    --include 874405 \
    --include 838948 \
    --exclude 17379835
```
