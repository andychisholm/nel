Model Building
===================

## Wikipedia

### Preparation
```
WK_DUMP_DATE=latest
WK_DUMP_FN=enwiki-$WK_DUMP_DATE-pages-articles.xml.bz2
WK_PLAIN_PATH=$WK_DUMP_FN\_extracted
WK_DOCREP_PATH=wikipedia.$WK_DUMP_DATE.dr
WK_SPLIT_PATH=wikipedia.$WK_DUMP_DATE.split
```
### Download
```
wget http://dumps.wikimedia.org/enwiki/latest/$WK_DUMP_FN
```
### Strip Wikimedia Markup
```
# tested with v2.6 of: https://github.com/bwbaugh/wikipedia-extractor
bzcat $WK_DUMP_FN|python WikiExtractor.py -cb 500k -l -s -o $WK_PLAIN_PATH
```

### Convert to Docrep
```
nel build-wikipedia-docrep $WK_PLAIN_PATH $WK_DOCREP_PATH

rm $WK_DUMP_FN
rm -r $WK_PLAIN_PATH
```

### Model Building
```
dr split --in-file $WK_DOCREP_PATH -t $WK_SPLIT_PATH/wikipedia.{n:03d}.dr k 1000

# EntityPrior, NameProbability and EntityOccurrence models
nel build-link-models $WK_SPLIT_PATH wikipedia

# Entity textual context models
nel build-context-models $WK_SPLIT_PATH wikipedia
```
