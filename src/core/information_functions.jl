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
# is in the individual information definition or discrete estimator files

# Convenience
information(est::ProbabilitiesEstimator, x) = information(Shannon(), est, x)
information(probs::Probabilities) = information(Shannon(), probs)


# Differential
"""
    information(est::DifferentialInfoEstimator, x) → h::Real

Approximate a **differential information measure** using the provided
[`DifferentialInfoEstimator`](@ref) and input data `x`.

The overwhelming majority of differential estimators estimate the [`Shannon`](@ref)
entropy. If the same estimator can estimate different information measures (e.g. it can
estimate both [`Shannon`](@ref) and [`Tsallis`](@ref)), then the information measure
is provided as an argument to the estimator itself.

See the [table of differential information measure estimators](@ref table_diff_ent_est)
in the docs for all differential information measure estimators.

Currently, unlike the discrete information measure this method doesn't involve explicitly
first computing a probability density function and then passing this density
to an information measure definition. But in the future, we want to establish a
`density` API similar to the [`probabilities`](@ref) API.

## Examples

A normal distribution has a base-e Shannon differential entropy of
`0.5*log(2π) + 0.5` nats.

```julia
est = Kraskov(k = 5, base = ℯ) # Base `ℯ` for nats.
h = information(est, randn(2_000_000))
abs(h - 0.5*log(2π) - 0.5) # ≈ 0.0001
```
"""
function information(est::DifferentialInfoEstimator, args...)
    throw(ArgumentError("`information` not implemented for $(nameof(typeof(est)))"))
end

# Dispatch for these functions is implemented in individual estimator files in
# `differential_info_estimators/`.

function information(::InformationMeasure, ::DifferentialInfoEstimator, args...)
    throw(ArgumentError(
        """`InformationMeasure` must be given as an argument to `est`, not to `information`.
        """
    ))
end
