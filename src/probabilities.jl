export ProbabilitiesEstimator, Probabilities
export probabilities, probabilities!
export probabilities_and_outcomes, outcomes
export total_outcomes
export missing_outcomes
export outcome_space

###########################################################################################
# Types
###########################################################################################
"""
    Probabilities <: AbstractVector
    Probabilities(x) → p

`Probabilities` is a simple wrapper around `x::AbstractVector{<:Real}` that ensures its
values sum to 1, so that `p` can be interpreted as probability mass function.
"""
struct Probabilities{T} <: AbstractVector{T}
    p::Vector{T}
    function Probabilities(x::AbstractVector{T}, normed = false) where T <: Real
        if !normed # `normed` is an internal argument that skips checking the sum.
            s = sum(x)
            if s ≠ 1
                x = x ./ s
            end
        end
        return new{T}(x)
    end
end
function Probabilities(x::AbstractVector{<:Integer})
    s = sum(x)
    return Probabilities(x ./ s, true)
end

# extend base Vector interface:
for f in (:length, :size, :eachindex, :eltype, :parent,
    :lastindex, :firstindex, :vec, :getindex, :iterate)
    @eval Base.$(f)(d::Probabilities, args...) = $(f)(d.p, args...)
end
Base.IteratorSize(::Probabilities) = Base.HasLength()
# Special extension due to the rules of the API
@inline Base.sum(::Probabilities{T}) where T = one(T)

"""
    ProbabilitiesEstimator

The supertype for all probabilities estimators.

In ComplexityMeasures.jl, probability distributions are estimated from data by defining a set of
possible outcomes ``\\Omega = \\{\\omega_1, \\omega_2, \\ldots, \\omega_L \\}``, and
assigning to each outcome ``\\omega_i`` a probability ``p(\\omega_i)``, such that
``\\sum_{i=1}^N p(\\omega_i) = 1``. It is the role of a [`ProbabilitiesEstimator`](@ref) to

1. Define ``\\Omega``, the "outcome space", which is the set of all possible outcomes over
    which probabilities are estimated. The cardinality of this set can be obtained using
    [`total_outcomes`](@ref).
2. Define how probabilities ``p_i = p(\\omega_i)`` are assigned to outcomes ``\\omega_i``.

In practice, probability estimation is done by calling [`probabilities`](@ref) with some
input data and one of the following probabilities estimators. The result is a
[`Probabilities`](@ref) `p` (`Vector`-like), where each element `p[i]` is the probability of
the outcome `ω[i]`. Use [`probabilities_and_outcomes`](@ref) if you need
both the probabilities and the outcomes, and use [`outcome_space`](@ref) to obtain
``\\Omega`` alone.
The element type of ``\\Omega`` varies between estimators, but it is guranteed to be
_hashable_. This allows for conveniently tracking the probability of a specific event
across experimental realizations, by using the outcome as a dictionary key and the
probability as the value for that key (or, alternatively, the key remains the outcome
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
a `Dataset`; see [Input data for ComplexityMeasures.jl](@ref).
Configuration options are always given as arguments to the chosen estimator.

To obtain the outcomes corresponding to these probabilities, use [`outcomes`](@ref).

Due to performance optimizations, whether the returned probablities
contain `0`s as entries or not depends on the estimator.
E.g., in [`ValueHistogram`](@ref) `0`s are skipped, while in
[`SymbolicPermutation`](@ref) `0` are not, because we get them for free.

    probabilities(x::Array_or_Dataset) → p::Probabilities

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

Return `probs, outs`, where `probs = probabilities(x, est)` and
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

Return a container containing all _possible_ outcomes of `est` for input `x`.

For some estimators the concrete outcome space is known without knowledge of input `x`,
in which case the function dispatches to `outcome_space(est)`.
In general it is recommended to use the 2-argument version irrespectively of estimator.
"""
function outcome_space(est::ProbabilitiesEstimator)
    error("""
    `outcome_space(est)` not implemented for estimator $(typeof(est)).
    Try calling `outcome_space(est, x)`, and if you get the same error, open an issue.
    """)
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
