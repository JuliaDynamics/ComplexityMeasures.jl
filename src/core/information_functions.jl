###########################################################################################
# Discrete
###########################################################################################

"""
    information([e::DiscreteInfoEstimator,] probs::Probabilities) → h::Real
    information([e::DiscreteInfoEstimator,] est::ProbabilitiesEstimator, x) → h::Real

Estimate a **discrete information measure**, using the estimator `est`, in one of two ways:

1. Directly from existing [`Probabilities`](@ref) `probs`.
2. From input data `x`, by first estimating a probability mass function using the provided
   [`ProbabilitiesEstimator`](@ref), and then computing the information measure from that
   mass fuction using the provided [`DiscreteInfoEstimator`](@ref).

Instead of providing a [`DiscreteInfoEstimator`](@ref), an
[`InformationMeasure`](@ref) can be given directly, in which case [`PlugIn`](@ref)
is used as the estimator. If `e` is not provided, [`Shannon`](@ref)`()` is used by default
in which case `information` returns the Shannon entropy.

Most discrete information measures have a well defined maximum value for a given probability
estimator. To obtain this value, call [`information_maximum`](@ref). Alternatively,
use [`information_normalized`](@ref) to obtain the normalized form of the measure (divided by the
maximum possible value).

## Examples

```julia
x = [rand(Bool) for _ in 1:10000] # coin toss
ps = probabilities(x) # gives about [0.5, 0.5] by definition
h = information(ps) # gives 1, about 1 bit by definition (Shannon entropy by default)
h = information(Shannon(), ps) # syntactically equivalent to the above
h = information(PlugIn(Shannon()), ps) # syntactically equivalent to the above
h = information(Shannon(), CountOccurrences(x), x) # syntactically equivalent to above
h = information(PlugIn(Renyi(2.0)), ps) # also gives 1, order `q` doesn't matter for coin toss
h = information(SymbolicPermutation(;m=3), x) # gives about 2, again by definition
```
"""
function information(e::Union{InformationMeasure, DiscreteInfoEstimator}, est::ProbabilitiesEstimator, x)
    ps = probabilities(est, x)
    return information(e, ps)
end
# dispatch for `information(e, ps::Probabilities)`
# is in the individual discrete entropy definition or estimator files

# Convenience
information(est::ProbabilitiesEstimator, x) = information(Shannon(), est, x)
information(probs::Probabilities) = information(Shannon(), probs)
information(e::PlugIn, args...) = information(e.definition, args...)
