# nnclj
Clojure implementation of [Neural Networks and Deep Learning](neuralnetworksanddeeplearning.com)

The original python2 implementation can be found [here](https://github.com/mnielsen/neural-networks-and-deep-learning)

The easiest way I could to get my hands on the MNIST dataset was from [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

# Thoughts
- lazy evaluation and easy concurrency with pmap is nice
- core.matrix with the vectorz implementation is fast
- however clojure in general (and vectorz) is built around doubles, pretty much impossible to use single-precision floats
