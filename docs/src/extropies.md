# [Extropies](@id extropies)

## Extropies API

The extropies API is defined by

- [`ExtropyDefinition`](@ref)
- [`extropy`](@ref)
- [`DiscreteExtropyEstimator`](@ref)

The extropy API is identical to the entropy API, except that the word "entropy" is
replaced by "extropy" everywhere. We do not yet offer differential extropy estimators,
although this will exist in the future (PRs are welcome!).

## Extropy definitions

```@docs
ExtropyDefinition
ShannonExtropy
RenyiExtropy
TsallisExtropy
```

## Discrete xntropy

```@docs
extropy(::ExtropyDefinition, ::ProbabilitiesEstimator, ::Any)
extropy_maximum
extropy_normalized
```

### Discrete extropy estimators

```@docs
DiscreteExtropyEstimator
MLExtropy
```
