nel
======================

Python based Named Entity Linking tool.

Implementation based on work described in *Entity Disambiguation with Web Links* ([pdf](http://aclweb.org/anthology/Q15-1011)).

### Linking Example

```python
from nel.doc import Doc
from nel.harness.format import markup_to_whitespace, doc_to_markup
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
        And includes coreference mentions like Obama.
        Hopefully we can link them to Wikipedia.
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

print doc_to_markup(doc, raw)
```

### Getting Started

Documentation is a **work in progress**, see the [install guide](docs/guide.md) to get started.

----------------
NEL is open-source software released under an MIT licence.
