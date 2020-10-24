# Symbolic

## Estimator

```@docs
SymbolicPermutation
```

## Symbolization utils

Some convenience functions for symbolization are provided.

```@docs 
symbolize
encode_motif
```

## Compute probabilities

```@docs 
probabilities(x::Dataset{N, T}, est::SymbolicPermutation) where {N, T}
```