# nnclj
Clojure implementation of [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com)

The original python2 implementation can be found [here](https://github.com/mnielsen/neural-networks-and-deep-learning)

The easiest way I could find to get my hands on the MNIST dataset was from [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

# Thoughts
- lazy evaluation and easy concurrency with pmap is nice
- core.matrix with the vectorz implementation is fast
- however vectorz is built around doubles
  - losing some speed since we can't take advantage of SIMD instructions with 2x parallelism for fp32 vs fp64
- should use CUDA anyways 
  - for my hardware we'd have theoretical ~10x performance on CUDA using fp32
- in general fp32 is awkward in clojure because builtin math operations convert to clojure.lang.Number which uses double-precision floats

- now I did a hy/pytorch translation which was pretty straightforward and almost verbatim from a pytorch example. 
- Pretty much boilerplate + (nn.Sequential (nn.Linear 784 100) (nn.Sigmoid) (nn.Linear 100 10) (nn.Sigmoid))
- the real clojure way would be to use libpython-clj
