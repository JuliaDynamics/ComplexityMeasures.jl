export ProbabilitiesEstimator, Probabilities
export probabilities, probabilities!
export probabilities_and_outcomes, outcomes
export total_outcomes
export missing_outcomes

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
The supertype for all probabilities estimators.

In Entropies.jl, probability distributions are estimated from data by defining a set of
possible outcomes ``\\Omega = \\{\\omega_1, \\omega_2, \\ldots, \\omega_L \\}``, and
assigning to each outcome ``\\omega_i`` a probability ``p(\\omega_i)``, such that
``\\sum_{i=1}^N p(\\omega_i) = 1``. It is the role a [`ProbabilitiesEstimator`](@ref) to

1. Define ``\\Omega``, an "outcome space", which is the set of possible outcomes over
    which probabilities are estimated. The cardinality of this set can be obtained using
    [`total_outcomes`](@ref).
2. Define how probabilities `p(ωᵢ)` are assigned to outcomes `ωᵢ`.

In practice, probability estimation is done by calling [`probabilities`](@ref) with some
input data and one of the following probabilities estimators. The result is a
[`Probabilities`](@ref) `p` (`Vector`-like), where each element `pᵢ` is the probability of
the outcome `ωᵢ`. Use [`probabilities_and_outcomes`](@ref) if you need
both the probabilities and the outcomes.

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

!!! note "A slight abuse of terminology"
    Formally, the outcome space is a *set*, but for consistency of the interface, we
    actually allow repeated elements among the outcomes for some estimators. This is
    the case for [`NaiveKernel`](@ref), for which probabilities are
    estimated for every input data point (even though some data points may be identical,
    they may be assigned different probabilities depending on their local neighborhood).
"""
abstract type ProbabilitiesEstimator end

###########################################################################################
# probabilities and outcomes
###########################################################################################
"""
    probabilities(x::Array_or_Dataset, est::ProbabilitiesEstimator) → p::Probabilities

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
function probabilities(x, est::ProbabilitiesEstimator)
    return probabilities_and_outcomes(x, est)[1]
end

"""
    probabilities_and_outcomes(x, est) → (probs, Ω::Vector)

Return `probs, Ω`, where `probs = probabilities(x, est)` and
`Ω[i]` is the outcome with probability `probs[i]`.
The element type of `Ω` depends on the estimator.
"""
function probabilities_and_outcomes(x, est::ProbabilitiesEstimator)
    error("`probabilities_and_outcomes` not implemented for estimator $(typeof(est)).")
end

"""
    outcomes(x, est::ProbabilitiesEstimator)
Return all (unique) outcomes contained in `x` according to the given estimator.
Equivalent with `probabilities_and_outcomes(x, est)[2]`, but for some estimators
it may be explicitly extended for better performance.
"""
function outcomes(x, est::ProbabilitiesEstimator)
    return probabilities_and_outcomes(x, est)[2]
end

"""
    probabilities!(s, args...)

Similar to `probabilities(args...)`, but allows pre-allocation of temporarily used
containers `s`.

Only works for certain estimators. See for example [`SymbolicPermutation`](@ref).
"""
function probabilities! end

###########################################################################################
# amount of outcomes
###########################################################################################
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
function total_outcomes(::Array_or_Dataset, est::ProbabilitiesEstimator)
    return total_outcomes(est)
end
function total_outcomes(est::ProbabilitiesEstimator)
    error("`total_outcomes` not known/implemented for estimator of type $(typeof(est)).")
end

"""
    missing_outcomes(x, est::ProbabilitiesEstimator) → n_missing::Int

Estimate a probability distribution for `x` using the given estimator, then count the number
of missing (i.e. zero-probability) outcomes.

Works for estimators that implement [`total_outcomes`](@ref).

See also: [`MissingDispersionPatterns`](@ref).
"""
function missing_outcomes end

function missing_outcomes(x::Array_or_Dataset, est::ProbabilitiesEstimator)
    probs = probabilities(x, est)
    L = total_outcomes(x, est)
    O = count(!iszero, probs)
    return L - O
end
