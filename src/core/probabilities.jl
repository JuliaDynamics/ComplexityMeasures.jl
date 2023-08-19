export ProbabilitiesEstimator, Probabilities
export probabilities, probabilities!
export probabilities_and_outcomes
export allprobabilities
export allprobabilities_and_outcomes
export missing_outcomes

###########################################################################################
# Types
###########################################################################################
"""
    Probabilities <: AbstractArray
    Probabilities(x) → p

`Probabilities` is a simple wrapper around `x::AbstractArray{<:Real, N}` that ensures its
values sum to 1, so that `p` can be interpreted as `N`-dimensional probability mass
function. In most use cases, `p` will be a vector. `p` behaves exactly like
its contained data `x` with respect to indexing and iteration.
"""
struct Probabilities{T, N} <: AbstractArray{T, N}
    p::Array{T, N}
    function Probabilities(x::AbstractArray{T, N}, normed = false) where {T <: Real, N}
        if !normed # `normed` is an internal argument that skips checking the sum.
            s = sum(x, dims = 1:N)
            if s ≠ 1
                x = x ./ s
            end
        end
        return new{T, N}(x)
    end
end
function Probabilities(x::AbstractArray{<:Integer, N}) where N
    s = sum(x)
    return Probabilities(x ./ s, true)
end

# extend base Array interface:
for f in (:length, :size, :eachindex, :eltype, :parent,
    :lastindex, :firstindex, :vec, :getindex, :iterate)
    @eval Base.$(f)(d::Probabilities{T, N}, args...) where {T, N} = $(f)(d.p, args...)
end

Base.IteratorSize(::Probabilities) = Base.HasLength()
# Special extension due to the rules of the API
@inline Base.sum(::Probabilities{T}) where T = one(T)

"""
    ProbabilitiesEstimator

The supertype for all probabilities estimators.

The role of the probabilities estimator is to convert (pseudo-)counts to probabilities.
Currently, the implementation of all probabilities estimators assume *finite* outcome
space with known cardinality (i.e. the user must *model*/*assume* what this outcome space
is). Therefore, `ProbabilitiesEstimator` accept an [`OutcomeSpace`](@ref) as the first
argument, which specifies the set of possible outcomes.

## Implementations

The default probabilities estimator is [`RelativeAmount`](@ref), which is compatible with any
[`OutcomeSpace`](@ref). The following estimators only support counting-based outcomes.

- [`Shrinkage`](@ref).
- [`BayesianRegularization`](@ref).
- [`AddConstant`](@ref).

## Description

In ComplexityMeasures.jl, probability mass functions
are estimated from data by defining a set of
possible outcomes ``\\Omega = \\{\\omega_1, \\omega_2, \\ldots, \\omega_L \\}`` (by
specifying an [`OutcomeSpace`](@ref)), and
assigning to each outcome ``\\omega_i`` a probability ``p(\\omega_i)``, such that
``\\sum_{i=1}^N p(\\omega_i) = 1`` (by specifying a `ProbabilitiesEstimator`).

## Used with

- [`probabilities`](@ref)/[`probabilities_and_outcomes`](@ref) for estimating a probability
    distribution over some [`OutcomeSpace`](@ref) from input data.
- [`allprobabilities`](@ref)/[`allprobabilities_and_outcomes`](@ref) for estimating a
    probability distribution from input data, guaranteeing inclusion of zero-probability
    outcomes.

The returned probabilities `p` are a [`Probabilities`](@ref) (`Vector`-like), where each
element `p[i]` is the probability of the outcome `ω[i]`. Using an [`OutcomeSpace`](@ref)
directly as input to [`probabilities`](@ref) or related functions is also possible, and is
equivalent to using the [`RelativeAmount`](@ref) estimator.
"""
abstract type ProbabilitiesEstimator end

# The following methods are defined over the outcome space, not the probabilities
# estimator, but we provide them for convenience.
outcome_space(est::ProbabilitiesEstimator) = est.outcomemodel
outcome_space(est::ProbabilitiesEstimator, x) = outcome_space(est.outcomemodel, x)
total_outcomes(est::ProbabilitiesEstimator) = total_outcomes(est.outcomemodel)
total_outcomes(est::ProbabilitiesEstimator, x) = total_outcomes(est.outcomemodel, x)
outcomes(est::ProbabilitiesEstimator, x) = outcomes(est.outcomemodel, x)
function allcounts_and_outcomes(est::ProbabilitiesEstimator, x)
    return allcounts_and_outcomes(est.outcomemodel, x)
end
allcounts(est::ProbabilitiesEstimator, x) = allcounts(est.outcomemodel, x)
function counts_and_outcomes(est::ProbabilitiesEstimator, x)
    return counts_and_outcomes(est.outcomemodel, x)
end
counts(est::ProbabilitiesEstimator, x) = counts(est.outcomemodel, x)

###########################################################################################
# probabilities and combo function
###########################################################################################
"""
    probabilities(o::OutcomeSpace, x::Array_or_SSSet) → p::Probabilities
    probabilities(est::ProbabilitiesEstimator, x::Array_or_SSSet) → p::Probabilities

Compute a probability distribution over the set of possible outcomes
defined by the [`OutcomeSpace`](@ref) `o`, given input data `x`, using maximum likelihood
probability estimation ([`RelativeAmount`](@ref)).

To use some other form of probabilities estimation than [`RelativeAmount`](@ref), use the second
signature. In this case, the outcome space is given as the first argument to a
[`ProbabilitiesEstimator`](@ref). Note that this only works for counting-based outcome
spaces (see [`OutcomeSpace`](@ref)'s docstring for list of compatible outcome spaces).

The input data is typically an `Array` or a `StateSpaceSet` (or `SSSet` for sort); see
[Input data for ComplexityMeasures.jl](@ref). Configuration options are always given as
arguments to the chosen outcome space.

To obtain the outcomes corresponding to the probabilities `p`, use [`outcomes`](@ref),
or [`probabilities_and_outcomes`](@ref), which return both the probabilities and the
outcomes together.

Due to performance optimizations, whether the returned probabilities
contain `0`s as entries or not depends on the outcome space.
E.g., in [`ValueHistogram`](@ref) `0`s are skipped, while in
[`PowerSpectrum`](@ref) `0` are not, because we get them for free.
Use [`allprobabilities`](@ref)/[`allprobabilities_and_outcomes`](@ref) to guarantee that
zero probabilities are also returned (may be slower).

## Examples

```julia
x = randn(500)
ps = probabilities(OrdinalPatterns(m = 3), x)
ps = probabilities(ValueHistogram(RectangularBinning(5)), x)
ps = probabilities(WaveletOverlap(), x)
```

The outcome space is here given as the first argument to `est`.

## Examples

```julia
x = randn(500)

# Syntactically equivalent to `probabilities(OrdinalPatterns(m = 3), x)`
ps = probabilities(RelativeAmount(OrdinalPatterns(m = 3)), x)

# Some more sophisticated ways of estimating probabilities:
ps = probabilities(BayesianRegularization(SymbolicPermutation(m = 3)), x)
ps = probabilities(Shrinkage(ValueHistogram(RectangularBinning(5))), x)

# Only the `RelativeAmount` estimator works with non-counting based outcome spaces,
# like for example `WaveletOverlap`.
ps = probabilities(RelativeAmount(WaveletOverlap()), x) # works
ps = probabilities(BayesianRegularization(WaveletOverlap()), x) # errors
```

    probabilities(x::Vector_or_SSSet) → p::Probabilities

Estimate probabilities by using directly counting the elements of `x`, assuming that
`Ω = sort(unique(x))`, i.e. that the outcome space is the unique elements of `x`.
This is mostly useful when `x` contains categorical data. It is syntactically equivalent
to `probabilities(RelativeAmount(CountOccurrences()), x)`.

See also: [`probabilities_and_outcomes`](@ref), [`allprobabilities`](@ref),
[`allprobabilities_and_outcomes`](@ref), [`Probabilities`](@ref),
[`ProbabilitiesEstimator`](@ref).
"""
function probabilities(est, x)
    return first(probabilities_and_outcomes(est, x))
end

"""
    probabilities_and_outcomes(est, x)

Return `probs, outs`, where `probs = probabilities(est, x)` and
`outs[i]` is the outcome with probability `probs[i]`.
The element type of `outs` depends on the estimator.
`outs` is a subset of the [`outcome_space`](@ref) of `est`.

See also [`outcomes`](@ref), [`total_outcomes`](@ref).
"""
function probabilities_and_outcomes(est::ProbabilitiesEstimator, x)
    error("`probabilities_and_outcomes` not implemented for estimator $(typeof(est)).")
end

"""
    probabilities!(s, args...)

Similar to `probabilities(args...)`, but allows pre-allocation of temporarily used
containers `s`.

Only works for certain estimators. See for example [`OrdinalPatterns`](@ref).
"""
function probabilities! end

###########################################################################################
# All probabilities
###########################################################################################
# This method is overriden by non-counting-based `OutcomeSpace`s. For counting-based
# `OutcomeSpace`s, we just utilize `counts_and_outcomes` to get the histogram, then
# normalize it when converting to `Probabilities`.
function probabilities_and_outcomes(o::OutcomeSpace, x)
    cts, outcomes = counts_and_outcomes(o, x)
    return Probabilities(cts), outcomes
end

"""
    allprobabilities(est::ProbabilitiesEstimator, x::Array_or_SSSet) → p
    allprobabilities(o::OutcomeSpace, x::Array_or_SSSet) → p

The same as [`probabilities`](@ref), but ensures that outcomes with `0` probability
are explicitly added in the returned vector. This means that `p[i]` is the probability
of `ospace[i]`, with `ospace = `[`outcome_space`](@ref)`(est, x)`.

This function is useful in cases where one wants to compare the probability mass functions
of two different input data `x, y` under the same estimator. E.g., to compute the
KL-divergence of the two PMFs assumes that the obey the same indexing. This is
not true for [`probabilities`](@ref) even with the same `est`, due to the skipping
of 0 entries, but it is true for [`allprobabilities`](@ref).
"""
function allprobabilities(est, x)
    return first(allprobabilities_and_outcomes(est, x))
end


# If an outcome space model is provided without specifying a probabilities estimator,
# then naive plug-in estimation is used (the `RelativeAmount` estimator). In the case of
# counting-based `OutcomeSpace`s, we explicitly count occurrences of each
# outcome in the encoded data. For non-counting-based `OutcomeSpace`s, we
# just fill in the non-considered outcomes with zero probabilities.

# Each `ProbabilitiesEstimator` subtype must extend this method explicitly.

"""
    allprobabilities_and_outcomes(o::OutcomeSpace, x::Array_or_SSSet) → (p, Ω)
    allprobabilities_and_outcomes(est::ProbabilitiesEstimator, x::Array_or_SSSet) → (p, Ω)

The same as [`allprobabilities`](@ref), but also returns the outcome space `Ω`.
"""
function allprobabilities_and_outcomes(o::OutcomeSpace, x::Array_or_SSSet)
    if is_counting_based(o)
        cts, outcomes = allcounts_and_outcomes(o, x)
        return Probabilities(cts), outcomes
    end

    probs, outs = probabilities_and_outcomes(o, x)
    ospace = vec(outcome_space(o, x))
    # We first utilize that the outcome space is sorted and sort probabilities
    # accordingly (just in case we have an estimator that is not sorted)
    s = sortperm(outs)
    sort!(outs)
    ps = probs[s]
    # we now iterate over possible outocomes;
    # if they exist in the observed outcomes, we push their corresponding probability
    # into the probabilities vector. If not, we push 0 into the probabilities vector!
    allprobs = eltype(ps)[]
    observed_index = 1 # index of observed outcomes
    for j in eachindex(ospace) # we made outcome space a vector on purpose
        ω = ospace[j]
        ωobs = outs[observed_index]
        if ω ≠ ωobs
            push!(allprobs, 0)
        else
            push!(allprobs, ps[observed_index])
            observed_index += 1
        end
        # Check whether we have exhausted observed outcomes
        if observed_index > length(outs)
            remaining_0s = length(ospace) - j
            append!(allprobs, zeros(remaining_0s))
            break
        end
    end
    return Probabilities(allprobs, true), ospace
end

"""
    missing_outcomes(o::OutcomeSpace, x; all = true) → n_missing::Int

Count the number of missing (i.e., zero-probability) outcomes
specified by `o`, given input data `x`, using [`RelativeAmount`](@ref)
probabilities estimation.

If `all == true`, then [`allprobabilities`](@ref) is used to compute the probabilities.
If `all == false`, then [`probabilities`](@ ref) is used to compute the probabilities.

This is syntactically equivalent to `missing_outcomes(RelativeAmount(o), x)`.

    missing_outcomes(est::ProbabilitiesEstimator, x) → n_missing::Int

Like above, but specifying a custom [`ProbabilitiesEstimator`](@ref).

See also: [`MissingDispersionPatterns`](@ref).
"""
function missing_outcomes(est::ProbabilitiesEstimator, x; all::Bool = true)
    if all
        probs = allprobabilities(est, x)
        L = length(probs)
    else
        probs = probabilities(est, x)
        L = total_outcomes(est, x)
    end
    O = count(!iszero, probs)
    return L - O
end
missing_outcomes(o::OutcomeSpace, x::Array_or_SSSet) = missing_outcomes(RelativeAmount(o), x)
