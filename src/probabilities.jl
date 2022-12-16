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

`Probabilities` is a simple wrapper around `AbstractVector` that ensures its values sum
to 1, so that `p` can be interpreted as probability distribution.
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
for f in (:length, :size, :eachindex, :eltype,
    :lastindex, :firstindex, :vec, :getindex, :iterate)
    @eval Base.$(f)(d::Probabilities, args...) = $(f)(d.p, args...)
end
Base.IteratorSize(::Probabilities) = Base.HasLength()
@inline Base.sum(::Probabilities{T}) where T = one(T)

"""
ProbabilitiesEstimator
The supertype for all probabilities estimators.

In Entropies.jl, probability distributions are estimated from data by defining a set of
possible outcomes ``\\Omega = \\{\\omega_1, \\omega_2, \\ldots, \\omega_L \\}``, and
assigning to each outcome ``\\omega_i`` a probability ``p(\\omega_i)``, such that
``\\sum_{i=1}^N p(\\omega_i) = 1``. It is the role a [`ProbabilitiesEstimator`](@ref) to

1. Define ``\\Omega``, the "outcome space", which is the set of all possible outcomes over
    which probabilities are estimated. The cardinality of this set can be obtained using
    [`total_outcomes`](@ref).
2. Define how probabilities ``p_i = p(\\omega_i)` are assigned to outcomes ``\\omega_i``.

In practice, probability estimation is done by calling [`probabilities`](@ref) with some
input data and one of the following probabilities estimators. The result is a
[`Probabilities`](@ref) `p` (`Vector`-like), where each element `p[i]` is the probability of
the outcome `ω[i]`. Use [`probabilities_and_outcomes`](@ref) if you need
both the probabilities and the outcomes and [`outcome_space`](@ref) to obtain ``\\Omega``.
The element type of ``\\Omega`` varies between estimators, but it is guranteed to be
_hashable_. This allows for conveniently tracking the probability of a specific event
across experimental realizations, by using the outcome as a dictionary key and the
probability as the value for that key (or, alternatively, the key remains the outcome
and one has a vector of probabilities, one for each experimental realization).

All currently implemented probability estimators are:

- [`CountOccurrences`](@ref).
- [`ValueHistogram`](@ref).
- [`TransferOperator`](@ref).
- [`Dispersion`](@ref).
- [`WaveletOverlap`](@ref).
- [`PowerSpectrum`](@ref).
- [`SymbolicPermutation`](@ref).
- [`SymbolicWeightedPermutation`](@ref).
- [`SymbolicAmplitudeAwarePermutation`](@ref).
- [`SpatialSymbolicPermutation`](@ref).
- [`NaiveKernel`](@ref).
"""
abstract type ProbabilitiesEstimator end

###########################################################################################
# probabilities and combo function
###########################################################################################
"""
    probabilities(est::ProbabilitiesEstimator, x::Array_or_Dataset) → p::Probabilities

Compute a probability distribution over the set of possible outcomes defined by the
probabilities estimator `est`, given input data `x`.
To obtain the outcomes use [`outcomes`](@ref).

The returned probabilities `p` may or may not be ordered, and may or may not
contain 0s; see the documentation of the individual estimators for more.
Configuration options are always given as arguments to the chosen estimator.
`x` is typically an `Array` or a `Dataset`; see [Input data for Entropies.jl](@ref).

    probabilities(x::Array_or_Dataset) → p::Probabilities

Estimate probabilities by directly counting the elements of `x`, assuming that
`Ω = sort(unique(x))`, i.e. that the outcome space is the unique elements of `x`.
This is mostly useful when `x` contains categorical or integer data.

See also: [`Probabilities`](@ref), [`ProbabilitiesEstimator`](@ref).
"""
function probabilities(est::ProbabilitiesEstimator, x)
    return probabilities_and_outcomes(est, x)[1]
end

"""
    probabilities_and_outcomes(x, est) → (probs, Ω::Vector)

Return `probs, Ω`, where `probs = probabilities(x, est)` and
`Ω[i]` is the outcome with probability `probs[i]`.
The element type of `Ω` depends on the estimator.

See also [`outcomes`](@ref), [`total_outcomes`](@ref), and [`outcome_space`](@ref).
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
    outcome_space([x,] est::ProbabilitiesEstimator) → Ω

Return a container (typically `Vector`) containing all _possible_ outcomes of `est`,
i.e., the outcome space `Ω`.
Only possible for estimators that implement [`total_outcomes`](@ref),
and similarly, for some estimators `x` is not needed. The _values_ of `x` are never needed;
but some times the type and dimensional layout of `x` is.
"""
function outcome_space(x, est::ProbabilitiesEstimator)
    outcome_space(est)
end
function outcome_space(est::ProbabilitiesEstimator)
    error(
        "`outcome_space(est)` not known/implemented for estimator $(typeof(est))."*
        "Try providing some input data, e.g. `outcomes_space(x, est)`."*
        "In some cases, this gives the dimensional layout/type information needed "*
        "to define the outcome space."
        )
end

"""
    total_outcomes([x::Array_or_Dataset,] est::ProbabilitiesEstimator) → Int

Return the size/cardinality of the outcome space ``\\Omega`` defined by the probabilities
estimator `est` imposed on the input data `x`.

For some estimators, the total number of outcomes is independent of `x`, in which case
the input `x` is ignored and the method `total_outcomes(est)` is called. If the total
number of states cannot be known a priori, an error is thrown. Primarily used in
[`entropy_normalized`](@ref).

## Examples

```jldoctest setup = :(using Entropies)
julia> est = SymbolicPermutation(m = 4);

julia> total_outcomes(rand(42), est) # same as `factorial(m)` for any `x`
24
```
"""
function total_outcomes(x, est::ProbabilitiesEstimator)
    return length(outcome_space(x, est))
end

"""
    missing_outcomes(x, est::ProbabilitiesEstimator) → n_missing::Int

Estimate a probability distribution for `x` using the given estimator, then count the number
of missing (i.e. zero-probability) outcomes.

Works for estimators that implement [`total_outcomes`](@ref).

See also: [`MissingDispersionPatterns`](@ref).
"""
function missing_outcomes(x::Array_or_Dataset, est::ProbabilitiesEstimator)
    probs = probabilities(x, est)
    L = total_outcomes(x, est)
    O = count(!iszero, probs)
    return L - O
end

"""
    outcomes(x, est::ProbabilitiesEstimator)
Return all (unique) outcomes contained in `x` according to the given estimator.
Equivalent with `probabilities_and_outcomes(x, est)[2]`, but for some estimators
it may be explicitly extended for better performance.
"""
function outcomes(x, est::ProbabilitiesEstimator)
    return probabilities_and_outcomes(est, x)[2]
end
