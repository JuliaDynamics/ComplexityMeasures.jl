# [Generalized entropies](@id generalized_entropies)

Computing entropy boils down to one thing: first estimating a probability distribution, then applying one of the generalized entropy formulas ([`entropy_renyi`](@ref) or [`entropy_tsallis`](@ref)) to these distributions. Thus, any of the implemented [probabilities estimators](@ref estimators) can be used to compute generalized entropies.

## Rényi (generalized) entropy

```@docs
Entropies.entropy_renyi
```

## Tsallis (generalized) entropy

```@docs
Entropies.entropy_tsallis
```
