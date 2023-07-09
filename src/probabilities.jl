export ProbabilitiesEstimator, Probabilities
export probabilities, probabilities!
export probabilities_and_outcomes, outcomes
export total_outcomes
export missing_outcomes
export outcome_space
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
    ProbabilitiesEstimator

The supertype for all probabilities estimators.

In ComplexityMeasures.jl, probability mass functions
are estimated from data by defining a set of
possible outcomes ``\\Omega = \\{\\omega_1, \\omega_2, \\ldots, \\omega_L \\}``, and
assigning to each outcome ``\\omega_i`` a probability ``p(\\omega_i)``, such that
``\\sum_{i=1}^N p(\\omega_i) = 1``. It is the role of a [`ProbabilitiesEstimator`](@ref) to

1. Define ``\\Omega``, the "outcome space", which is the set of all possible outcomes over
    which probabilities are estimated.
2. Define how probabilities ``p_i = p(\\omega_i)`` are assigned to outcomes ``\\omega_i``
   give input data.

In practice, probability estimation is done by calling [`probabilities`](@ref) with some
input data and one of the implemented probabilities estimators. The result is a
[`Probabilities`](@ref) `p` (`Vector`-like), where each element `p[i]` is the probability of
the outcome `ω[i]`. Use [`probabilities_and_outcomes`](@ref) if you need
both the probabilities and the outcomes, and use [`outcome_space`](@ref) to obtain
``\\Omega`` alone.  The cardinality of ``\\Omega`` can be obtained using
[`total_outcomes`](@ref).

The element type of ``\\Omega`` varies between estimators, but it is guaranteed to be
_hashable_ and _sortable_. This allows for conveniently tracking the probability of a
specific event across experimental realizations, by using the outcome as a dictionary key
and the probability as the value for that key (or, alternatively, the key remains the outcome
and one has a vector of probabilities, one for each experimental realization).

Some estimators can deduce ``\\Omega`` without knowledge of the input, such as
[`SymbolicPermutation`](@ref). For others, knowledge of input is necessary for concretely
specifying ``\\Omega``, such as [`ValueHistogram`](@ref) with [`RectangularBinning`](@ref).
This only matters for the functions [`outcome_space`](@ref) and [`total_outcomes`](@ref).

All currently implemented probability estimators are listed in a nice table in the
[probabilities estimators](@ref probabilities_estimators) section of the online documentation.
"""
abstract type ProbabilitiesEstimator end

###########################################################################################
# probabilities and combo function
###########################################################################################
"""
    probabilities(est::ProbabilitiesEstimator, x::Array_or_Dataset) → p::Probabilities

Compute a probability distribution over the set of possible outcomes defined by the
probabilities estimator `est`, given input data `x`, which is typically an `Array` or
a `StateSpaceSet`; see [Input data for ComplexityMeasures.jl](@ref).
Configuration options are always given as arguments to the chosen estimator.

To obtain the outcomes corresponding to these probabilities, use [`outcomes`](@ref).

Due to performance optimizations, whether the returned probablities
contain `0`s as entries or not depends on the estimator.
E.g., in [`ValueHistogram`](@ref) `0`s are skipped, while in
[`PowerSpectrum`](@ref) `0` are not, because we get them for free.
Use the function [`allprobabilities`](@ref) for a version with all `0` entries
that ensures that given an `est`, the indices of `p` will be independent
of the input data `x`.

    probabilities(x::Vector_or_Dataset) → p::Probabilities

Estimate probabilities by directly counting the elements of `x`, assuming that
`Ω = sort(unique(x))`, i.e. that the outcome space is the unique elements of `x`.
This is mostly useful when `x` contains categorical data.

See also: [`Probabilities`](@ref), [`ProbabilitiesEstimator`](@ref).
"""
function probabilities(est::ProbabilitiesEstimator, x)
    return probabilities_and_outcomes(est, x)[1]
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
# Outcome space
###########################################################################################
"""
    outcome_space(est::ProbabilitiesEstimator, x) → Ω

Return a sorted container containing all _possible_ outcomes of `est` for input `x`.

For some estimators the concrete outcome space is known without knowledge of input `x`,
in which case the function dispatches to `outcome_space(est)`.
In general it is recommended to use the 2-argument version irrespectively of estimator.
"""
function outcome_space(est::ProbabilitiesEstimator)
    error(ErrorException("""
    `outcome_space(est)` not implemented for estimator $(typeof(est)).
    Try calling `outcome_space(est, input_data)`, and if you get the same error, open an issue.
    """))
end
outcome_space(est::ProbabilitiesEstimator, x) = outcome_space(est)

"""
    total_outcomes(est::ProbabilitiesEstimator, x)

Return the length (cardinality) of the outcome space ``\\Omega`` of `est`.

For some estimators the concrete outcome space is known without knowledge of input `x`,
in which case the function dispatches to `total_outcomes(est)`.
In general it is recommended to use the 2-argument version irrespectively of estimator.
"""
total_outcomes(est::ProbabilitiesEstimator, x) = length(outcome_space(est, x))
total_outcomes(est::ProbabilitiesEstimator) = length(outcome_space(est))

"""
    missing_outcomes(est::ProbabilitiesEstimator, x) → n_missing::Int

Estimate a probability distribution for `x` using the given estimator, then count the number
of missing (i.e. zero-probability) outcomes.

See also: [`MissingDispersionPatterns`](@ref).
"""
function missing_outcomes(est::ProbabilitiesEstimator, x::Array_or_Dataset)
    probs = probabilities(est, x)
    L = total_outcomes(est, x)
    O = count(!iszero, probs)
    return L - O
end

"""
    outcomes(est::ProbabilitiesEstimator, x)
Return all (unique) outcomes contained in `x` according to the given estimator.
Equivalent with `probabilities_and_outcomes(x, est)[2]`, but for some estimators
it may be explicitly extended for better performance.
"""
function outcomes(est::ProbabilitiesEstimator, x)
    return probabilities_and_outcomes(est, x)[2]
end

###########################################################################################
# All probabilities
###########################################################################################
"""
    allprobabilities(est::ProbabilitiesEstimator, x::Array_or_Dataset) → p

Same as [`probabilities`](@ref), however ensures that outcomes with `0` probability
are explicitly added in the returned vector. This means that `p[i]` is the probability
of `ospace[i]`, with `ospace = `[`outcome_space`](@ref)`(est, x)`.

This function is useful in cases where one wants to compare the probability mass functions
of two different input data `x, y` under the same estimator. E.g., to compute the
KL-divergence of the two PMFs assumes that the obey the same indexing. This is
not true for [`probabilities`](@ref) even with the same `est`, due to the skipping
of 0 entries, but it is true for [`allprobabilities`](@ref).
"""
function allprobabilities(est::ProbabilitiesEstimator, x::Array_or_Dataset)
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