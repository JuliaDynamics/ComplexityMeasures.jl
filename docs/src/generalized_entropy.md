# Generalized entropy

Generalized entropy is a property of probability distributions. We provide the following
interfaces for computing generalized entropy, either directly on pre-computed distributions,
indirectly by first applying some `ProbabilityEstimator`, or directly using some 
`EntropyEstimator`. Check the docstrings for individual estimators to see which methods
work on which kinds of data.

```@docs
Entropies.genentropy(p::Probabilities)
```
