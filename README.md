# RegionEmbedding
Mxnet implementation of ICLR 2018 paper: [A new method of region embedding for text classification](https://openreview.net/forum?id=BkSDMA36Z).

Official implementation in [TensorFlow](https://github.com/text-representation/local-context-unit).

## 0.Notes

- I implemented both Word-Context and Context-Word Region Embedding in the paper.
- Please see the original papar about the datasets and pre-pocessing.
## 1.Requirements

- Python2 or Python3
- Mxnet 1.2.0
## 2.Results

|Datasets| Accuracy(%)|Best Epoch|
| :-- | :--: | :--: |
|Yahoo Answer|73.07(73.7)|2|
|Amazon Polarity|95.24(95.1)|3|
|Amazon Full|61.58(60.9)|2|
|Ag news| 92.96(92.8)|6|
|DBPedia|98.91(98.9)|4|
|Yelp Full| 64.98(64.9)|3|

note: The accuracy in brackets are results reported in the original paper.
