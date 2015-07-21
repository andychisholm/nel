# Offline Linking

## Basic
```bash
CONLL_DATA_PATH=conll.txt
GOLD_OUTPUT=conll_dev.gold.tsv
SYSTEM_OUTPUT=conll_dev.sys.tsv

# read in and preprocess documents from the corpus, storing them in the nel datastore
nel prepare-corpora conll conll $CONLL_DATA_PATH

# extract an EntityProbability feature using the 'wikipedia' derived model
nel extract-feature --corpus conll EntityProbability wikipedia

if [ -e "$GOLD_OUTPUT" ]; then
    # generate the gold-standard output against which we will evaluate this feature
    nel batch-link --corpus conll --tag dev --fmt neleval > $GOLD_OUTPUT
fi

# generate system output using entity prior as a ranker for ambiguous mentions
nel batch-link --corpus conll --tag dev --fmt neleval --ranker EntityProbability[wikipedia] > $SYSTEM_OUTPUT

# evaluate our system in terms of the strong-link-match metric with the neleval tool
neleval evaluate -m strong_link_match -f tab -g $GOLD_OUTPUT $SYSTEM_OUTPUT
```

## Supervised Feature Combination
```bash
# extract an EntityProbability feature using the 'wikipedia' derived model
nel extract-feature --corpus conll EntityProbability wikipedia

# extract a NameProbability feature using the 'wikipedia' derived model
nel extract-feature --corpus conll EntityProbability wikipedia

# train a ranking classifier over 'train' documents to combine these two features
nel train
    combined_pm \
    --corpus conll \
    --tag train \
    --feature EntityProbability[wikipedia] \
    --feature NameProbability[wikipedia]

# extract scores from this ranking classifier over the corpus
nel extract-feature --corpus conll ClassifierScore combined_pm

# generate system output using these scores as a ranker for ambiguous mentions
nel batch-link --corpus conll --tag dev --fmt neleval --ranker ClassifierScore[combined_pm] > $SYSTEM_OUTPUT
```

## Re-ranking with Coherence
```bash
# given an initial ranking, we can extract coherence features based on entity co-mention counts in wikipedia
nel extract-feature --corpus conll MeanConditionalProbability ClassifierScore[combined_pm] wikipedia

# train a ranking classifier including both base features and coherence
nel train
    reranker \
    --corpus conll \
    --tag train \
    --feature EntityProbability[wikipedia] \
    --feature NameProbability[wikipedia] \
    --feature ClassifierScore[combined_pm] \
    --feature MeanConditionalProbability[ClassifierScore[combined_pm]]

# extract scores from the re-ranking classifier over the corpus
nel extract-feature --corpus conll ClassifierScore reranker

# generate system output using these scores as a ranker for ambiguous mentions
nel batch-link --corpus conll --tag dev --fmt neleval --ranker ClassifierScore[reranker] > $SYSTEM_OUTPUT
```

# Running from Python

```python
from nel.doc import Doc
from nel.harness.format import markup_to_whitespace, inject_html_links
from nel.process.process import Pipeline
from nel.process.tokenise import RegexTokeniser, TOKEN_RE
from nel.process.tag import StanfordTagger, CandidateGenerator
from nel.features.probability import EntityProbability
from nel.process.resolve import FeatureRankResolver, GreedyOverlapResolver

raw = """
<html>
  <body>
    Sample document with html <a href='#'>markup</a>.
    It mentions a prominent entity like Barack Obama.
    And contains coreferential mentions like Obama.
    NEL can disambiguate these against Wikipedia and annotate the original document with links.
  </body>
</html>
""".decode('utf-8')

# create a plaintext document using a whitespace converter to preserve offsets
processed = markup_to_whitespace(raw)

# create a nel document containing the processed text
doc = Doc(doc_id='test', text=processed)

linker = Pipeline([
  # tokenise the document using a simple regex
  RegexTokeniser(TOKEN_RE),

  # tag the document using a hosted NER tagging service
  StanfordTagger('127.0.0.1', 1447),

  # generate wikipedia candidates for each mention from an alias set
  CandidateGenerator('wikipedia'),

  # extract a prior feature for each candidate based on wikipedia in-link counts
  EntityProbability('wikipedia'),

  # resolve each mention to the entity with the highest entity prior
  FeatureRankResolver('EntityProbability[wikipedia]'),

  # some taggers produce overlapping mentions, we resolve these by taking the mention with the highest score
  GreedyOverlapResolver('EntityProbability[wikipedia]')
])

doc = linker(doc)

print inject_html_links(raw, doc)
```

### Output
```
<html>
  <body>
    Sample document with html <a href='#'>markup</a>.
    It mentions a prominent entity like <a href="https://en.wikipedia.org/wiki/Barack_Obama">Barack Obama</a>.
    And contains coreferential mentions like <a href="https://en.wikipedia.org/wiki/Barack_Obama">Obama</a>.
    NEL can disambiguate these against <a href="https://en.wikipedia.org/wiki/Wikipedia">Wikipedia</a> and annotate the original document with links.
  </body>
</html>
```
