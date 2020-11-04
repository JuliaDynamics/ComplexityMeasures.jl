# Generalized entropy

## For probability distributions

Generalized entropy is a property of probability distributions.

```@docs
Entropies.genentropy(α::Real, p::AbstractArray{T}; base = Base.MathConstants.e) where {T <: Real}
```

## For real data (ordered sequences, time series)

The method above only works when you actually have access to a probability distribution.
In most cases, probability distributions have to be estimated from data.

Currently, we implement the following probability estimators:

- [`CountOccurrences`](@ref)
- [`VisitationFrequency`](@ref)
- [`SymbolicPermutation`](@ref)
- [`SymbolicWeightedPermutation`](@ref)
- [`SymbolicAmplitudeAwarePermutation`](@ref)

### Getting the distributions

Distributions can be obtained directly for dataset `x` using the signature

```julia
probabilities(x, estimator)
```

### Computing the entropy

The syntax for using the different estimators to compute generalized entropy are as follows.

```@docs
Entropies.genentropy(::AbstractDataset)
```
