export alphabet_length

"""
    alphabet_length(estimator) → Union{Int, Nothing}

Returns the total number of possible symbols/states implied by `estimator`.
If the total number of states cannot be known a priori, `nothing` is returned.
Primarily used for normalization of entropy values computed using symbolic estimators.

## Examples

```jldoctest setup = :(using Entropies)
julia> est = SymbolicPermutation(m = 4)
SymbolicPermutation{typeof(Entropies.isless_rand)}(1, 4, Entropies.isless_rand)

julia> alphabet_length(est)
24
```
"""
function alphabet_length end

alphabet_length(est::SymbolicPermutation)::Int = factorial(est.m)
alphabet_length(est::SymbolicWeightedPermutation)::Int = factorial(est.m)
alphabet_length(est::SymbolicAmplitudeAwarePermutation)::Int = factorial(est.m)
alphabet_length(est::Dispersion)::Int = est.symbolization.c ^ est.m
