# Building a Linker

## Pipeline Config

```javascript
[{
    "type": "Tagger",
    "name": "CRFTagger",
    "params": {
        "model_tag": "crf"
    }
}, {
    "type": "MentionClusterer",
    "name": "SpanOverlap",
    "params": {}
}, {
    "type": "CandidateGenerator",
    "name": "NameCounts",
    "params": {
        "name_model_tag": "wikipedia",
        "limit": 15
    }
}, {
    "type": "Feature",
    "name": "EntityProbability",
    "params": {
        "entity_model_tag": "wikipedia"
    }
}, {
    "type": "Feature",
    "name": "NameProbability",
    "params": {
        "name_model_tag": "wikipedia"
    }
}, {
    "type": "Feature",
    "name": "ClassifierScore",
    "params": {
        "classifier": "ranker"
    }
}, {
    "type": "Resolver",
    "name": "FeatureRankResolver",
    "params": {
        "ranking_feature": "ClassifierScore[ranker]"
    }
}]
```
