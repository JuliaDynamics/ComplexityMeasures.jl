# Probabilities

```@docs
probabilities
probabilities!
```

Specifics on how probabilities may be estimated follow.

```@docs
probabilities(x::Dataset, est::SymbolicPermutation)
probabilities(x::Dataset, est::SymbolicWeightedPermutation)
probabilities(x::Dataset, est::SymbolicAmplitudeAwarePermutation)
probabilities(x::Dataset, est::VisitationFrequency)
probabilities(x::AbstractVector{<:Real}, est::TimeScaleMODWT)
```
