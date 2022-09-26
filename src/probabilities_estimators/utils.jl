export alphabet_length

"""
    alphabet_length(estimator) → Int

Returns the total number of possible symbols/states implied by `estimator`.
If the total number of states cannot be known a priori, an error is thrown.
Primarily used for normalization of entropy values computed using symbolic estimators.

## Examples

```jldoctest setup = :(using Entropies)
julia> est = SymbolicPermutation(m = 4)
SymbolicPermutation{typeof(Entropies.isless_rand)}(1, 4, Entropies.isless_rand)

julia> alphabet_length(est)
24
```
"""
alphabet_length(est) =
    throw(error("alphabet_length not implemented for estimator of type $(typeof(est))"))
