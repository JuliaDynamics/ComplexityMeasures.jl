export ProbabilitiesEstimator, Probabilities
export probabilities, probabilities!
export probabilities_and_outcomes
export allprobabilities

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
    ProbabilitiesEstimator{<:Outcome space}

The supertype for all probabilities estimators.

All subtypes takes an [`OutcomeSpaceModel`](@ref) as its first input,
which defines a set of possible outcomes over which observed frequencies are counted.
It is the role of a `ProbabilitiesEstimator` to convert these frequencies to
probabilities.

## implementations

- [`MLEP`](@ref).


## Description

In ComplexityMeasures.jl, probability mass functions
are estimated from data by defining a set of
possible outcomes ``\\Omega = \\{\\omega_1, \\omega_2, \\ldots, \\omega_L \\}``, and
assigning to each outcome ``\\omega_i`` a probability ``p(\\omega_i)``, such that
``\\sum_{i=1}^N p(\\omega_i) = 1``.

In practice, probability estimation is done by calling [`probabilities`](@ref) with some
input data and one of the implemented probabilities estimators. The result is a
[`Probabilities`](@ref) `p` (`Vector`-like), where each element `p[i]` is the probability of
the outcome `ω[i]`. Use [`probabilities_and_outcomes`](@ref) if you need
both the probabilities and the outcomes, and use [`outcome_space`](@ref) to obtain
``\\Omega`` alone.  The cardinality of ``\\Omega`` can be obtained using
[`total_outcomes`](@ref).

Currently, all probabilities estimators estimate the *plug-in* (or maximimum likelihood)
estimates of probabilities.
"""
abstract type ProbabilitiesEstimator end

# The following methods are defined over the outcome space, not the probabilities
# estimator, but we provide them for convenience.
outcome_space(est::ProbabilitiesEstimator) = est.outcome_space
outcome_space(est::ProbabilitiesEstimator, x) = outcome_space(est.outcome_space, x)
total_outcomes(esc::ProbabilitiesEstimator) = total_outcomes(est.outcome_space)
total_outcomes(esc::ProbabilitiesEstimator, x) = total_outcomes(est.outcome_space, x)
outcomes(est::ProbabilitiesEstimator, x) = outcomes(est.outcome_space, x)

###########################################################################################
# probabilities and combo function
###########################################################################################
"""
    probabilities(o::OutcomeSpaceModel, x::Array_or_SSSet) → p::Probabilities

Syntactically equivalent to using `probabilities(MLEP(o), x)` (see below).

    probabilities(est::ProbabilitiesEstimator, x::Array_or_SSSet) → p::Probabilities

Compute a probability distribution over the set of possible outcomes defined by the
probabilities estimator, given input data `x`, which is typically an
`Array` or a `StateSpaceSet` (or `SSSet` for sort); see [Input data for
ComplexityMeasures.jl](@ref). Configuration options are always given as arguments to the
chosen estimator.

To obtain the outcomes corresponding to these probabilities, use [`outcomes`](@ref).

Due to performance optimizations, whether the returned probablities
contain `0`s as entries or not depends on the estimator.
E.g., in [`ValueHistogram`](@ref) `0`s are skipped, while in
[`PowerSpectrum`](@ref) `0` are not, because we get them for free.
Use the function [`allprobabilities`](@ref) for a version with all `0` entries
that ensures that given an `est`, the indices of `p` will be independent
of the input data `x`.

    probabilities(x::Vector_or_SSSet) → p::Probabilities

Estimate probabilities by using directly counting the elements of `x`, assuming that
`Ω = sort(unique(x))`, i.e. that the outcome space is the unique elements of `x`.
This is mostly useful when `x` contains categorical data. It is syntactically equivalent
to using a [`MLEP`](@ref) estimator.

See also: [`Probabilities`](@ref), [`ProbabilitiesEstimator`](@ref).
"""
function probabilities(est::ProbabilitiesEstimator, x)
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

Only works for certain estimators. See for example [`SymbolicPermutation`](@ref).
"""
function probabilities! end


###########################################################################################
# All probabilities
###########################################################################################
"""
    allprobabilities(o::OutcomeSpace, x::Array_or_SSSet) → p

Same as [`probabilities`](@ref), however ensures that outcomes with `0` probability
are explicitly added in the returned vector. This means that `p[i]` is the probability
of `ospace[i]`, with `ospace = `[`outcome_space`](@ref)`(est, x)`.

This function is useful in cases where one wants to compare the probability mass functions
of two different input data `x, y` under the same estimator. E.g., to compute the
KL-divergence of the two PMFs assumes that the obey the same indexing. This is
not true for [`probabilities`](@ref) even with the same `est`, due to the skipping
of 0 entries, but it is true for [`allprobabilities`](@ref).
"""
function allprobabilities(o::OutcomeSpaceModel, x::Array_or_SSSet)
    probs, outs = probabilities_and_outcomes(est, x)
    ospace = vec(outcome_space(est, x))
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
    return Probabilities(allprobs, true)
end
