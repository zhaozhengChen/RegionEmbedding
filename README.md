# RegionEmbedding
Mxnet implementation of ICLR 2018 paper: [A new method of region embedding for text classification](https://openreview.net/forum?id=BkSDMA36Z).

Official implementation in [TensorFlow](https://github.com/text-representation/local-context-unit).

## 0.Notes

- I implemented both Word-Context and Context-Word Region Embedding in the paper.
- Please see the original papar about the datasets and pre-pocessing.
- All the hyper-parameters I used are copied from the official implementation.
## 1.Requirements

- Python2 or Python3
- Mxnet 1.2.0
## 2.Results

To be updated

|Datasets| Accuracy(%)<br>WordContext|Best Epoch<br>WordContext|Accuracy(%)<br>ContextWord|Best Epoch<br>ContextWord|
| :-- | :--: | :--: | :--: | :--: |
|Yahoo Answer|73.07(73.7)|2|73.42(73.4)|3|
|Amazon Polarity|95.27(95.1)|2|--(95.3)|--|
|Amazon Full|61.58(60.9)|2|61.59(60.8)|2|
|Ag news| 92.96(92.8)|6|92.89(92.8)|8|
|DBPedia|98.91(98.9)|4|98.88(98.9)|3|
|Yelp Full| 64.98(64.9)|3|64.94(64.5)|2|

Note: 
- The accuracy in brackets are results reported in the original paper.
