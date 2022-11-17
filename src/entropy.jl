export Entropy, EntropyEstimator
export entropy, entropy_maximum, entropy_normalized, entropy!

abstract type AbstractEntropy end

"""
    Entropy

`Entropy` is the supertype of all (generalized) entropies, and currently implemented
entropy types are:

- [`Renyi`](@ref).
- [`Tsallis`](@ref).
- [`Shannon`](@ref), which is a subcase of the above two in the limit `q → 1`.
- [`Curado`](@ref).
- [`StretchedExponential`](@ref).

These entropy types are given as inputs to [`entropy`](@ref) and [`entropy_normalized`].

## Description

Mathematically speaking, generalized entropies are just nonnegative functions of
probability distributions that verify certain (entropy-type-dependent) axioms.
Amigó et al., 2018's
[summary paper](https://www.mdpi.com/1099-4300/20/11/813) gives a nice overview.

[Amigó2018]:
    Amigó, J. M., Balogh, S. G., & Hernández, S. (2018). A brief review of
    generalized entropies. Entropy, 20(11), 813.
"""
abstract type Entropy <: AbstractEntropy end

"""
    EntropyEstimator

The supertype of all entropy estimators.

These estimators compute some [`Entropy`](@ref) in various ways that doesn't involve
explicitly estimating a probability distribution. Currently implemented estimators are:

- [`KozachenkoLeonenko`](@ref)
- [`Kraskov`](@ref)
- [`Zhu`](@ref)
- [`ZhuSingh`](@ref)
- [`Vasicek`](@ref)
- [`Ebrahimi`](@ref)
- [`Correa`](@ref)
- [`AlizadehArghami`](@ref)

For example, [`entropy`](@ref)`(Shannon(), Kraskov(), x)` computes the Shannon entropy
of the input data `x` using the [`Kraskov`](@ref) `k`-th nearest neighbor estimator.
"""
abstract type EntropyEstimator <: AbstractEntropy end

###########################################################################################
# API: entropy from probabilities
###########################################################################################
# Notice that StatsBase.jl exports `entropy` and Wavelets.jl exports `Entropy`.
"""
    entropy([e::Entropy,] probs::Probabilities) → h::Real ∈ [0, ∞)
    entropy([e::Entropy,] est::ProbabilitiesEstimator, x) → h::Real ∈ [0, ∞)
    entropy([e::Entropy,] est::EntropyEstimator, x) → h::Real ∈ [0, ∞)

Compute `h`, a (generalized) [`Entropy`](@ref) of type `e`, in one of three ways:

1. Directly from existing [`Probabilities`](@ref) `probs`.
2. From input data `x`, by first estimating a probability distribution using the provided
    [`ProbabilitiesEstimator`](@ref), then computing entropy from that distribution.
    In fact, the second method is just a 2-lines-of-code wrapper that calls
    [`probabilities`](@ref) and gives the result to the first method.
3. From input data `x`, by using a dedicated [`EntropyEstimator`](@ref) that computes
    entropy in a way that doesn't involve explicitly computing probabilities first.

The entropy (first argument) is optional. When `est` is a probability estimator,
`Shannon()` is used by default. When `est` is a dedicated entropy estimator,
the default entropy type is inferred from the estimator (e.g. [`Kraskov`](@ref)
estimates the [`Shannon`](@ref) entropy).

## Input data

`x` is typically an `Array` or a `Dataset`, see [Input data for Entropies.jl](@ref).

## Maximum entropy and normalized entropy

All entropies `e` have a well defined maximum value for a given probability estimator.
To obtain this value one only needs to call the [`entropy_maximum`](@ref) function with the
chosen entropy type and probability estimator. Or, one can use [`entropy_normalized`](@ref)
to obtain the normalized form of the entropy (divided by the maximum).

## Examples

```julia
x = [rand(Bool) for _ in 1:10000] # coin toss
ps = probabilities(x) # gives about [0.5, 0.5] by definition
h = entropy(ps) # gives 1, about 1 bit by definition
h = entropy(Shannon(), ps) # syntactically equivalent to above
h = entropy(Shannon(), CountOccurrences(), x) # syntactically equivalent to above
h = entropy(SymbolicPermutation(;m=3), x) # gives about 2, again by definition
h = entropy(Renyi(2.0), ps) # also gives 1, order `q` doesn't matter for coin toss
```
"""
function entropy(e::Entropy, est::ProbabilitiesEstimator, x)
    ps = probabilities(est, x)
    return entropy(e, ps)
end

# Convenience
entropy(est::ProbabilitiesEstimator, x::Array_or_Dataset) = entropy(Shannon(), est, x)
entropy(probs::Probabilities) = entropy(Shannon(), probs)

"""
    entropy!(s, [e::Entropy,] x, est::ProbabilitiesEstimator)

Similar to `probabilities!`, this is an in-place version of [`entropy`](@ref) that allows
pre-allocation of temporarily used containers.

The entropy (second argument) is optional: if not given, `Shannon()` is used instead.

Only works for certain estimators. See for example [`SymbolicPermutation`](@ref).
"""
function entropy!(s::AbstractVector{Int}, e::Entropy, est::ProbabilitiesEstimator, x)
    probs = probabilities!(s, est, x)
    entropy(e, probs)
end

entropy!(s::AbstractVector{Int}, est::ProbabilitiesEstimator, x) =
    entropy!(s, Shannon(), est, x)

###########################################################################################
# API: entropy from entropy estimators
###########################################################################################
# Dispatch for these functions is implemented in individual estimator files in
# `entropies/estimators/`.
function entropy(e::Entropy, est::EntropyEstimator, x) end
entropy(est::EntropyEstimator, ::Probabilities) =
    error("Entropy estimators like $(nameof(typeof(est))) are not called with probabilities.")
entropy(e::Entropy, est::EntropyEstimator, ::Probabilities) =
    error("Entropy estimators like $(nameof(typeof(est))) are not called with probabilities.")

#
entropy(e::Entropy, est::EntropyEstimator, x::AbstractVector{<:Real}) =
    entropy(e, est, Dataset(x))

###########################################################################################
# Normalize API
###########################################################################################
"""
    entropy_maximum(e::Entropy, est::ProbabilitiesEstimator, x) → m::Real

Return the maximum value `m` of the given entropy type based on the given estimator
and the given input `x` (whose values are not important, but layout and type are).

This function only works if the maximum value is deducable, which is possible only
when the estimator has a known [`total_outcomes`](@ref).

    entropy_maximum(e::Entropy, L::Int) → m::Real

Alternatively, compute the maximum entropy from the number of total outcomes `L` directly.
"""
function entropy_maximum(e::Entropy, est::ProbabilitiesEstimator, x)
    L = total_outcomes(x, est)
    return entropy_maximum(e, L)
end
function entropy_maximum(e::Entropy, ::Int)
    error("not implemented for entropy type $(nameof(typeof(e))).")
end

"""
    entropy_normalized([e::Entropy,] est::ProbabilitiesEstimator, x) → h̃ ∈ [0, 1]

Return h̃, the normalized entropy of `x`, i.e. the value of [`entropy`](@ref) divided
by the maximum value for `e`, according to the given probabilities estimator.
If `e` is not given, it defaults to `Shannon()`.

Notice that unlike for [`entropy`](@ref), here there is no method
`entropy_normalized(e::Entropy, probs::Probabilities)` because there is no way to know
the amount of _possible_ events (i.e., the [`total_outcomes`](@ref)) from `probs`.
"""
function entropy_normalized(e::Entropy, est::ProbabilitiesEstimator, x)
    return entropy(e, est, x) / entropy_maximum(e, est, x)
end
function entropy_normalized(est::ProbabilitiesEstimator, x::Array_or_Dataset)
    return entropy_normalized(Shannon(), est, x)
end

###########################################################################################
# Utils
###########################################################################################
"""
    log_with_base(base) → f

Return a function that computes the logarithm at a given base.
This definitely increases accuracy, and probably also performance.
"""
function log_with_base(base)
    if base == 2
        log2
    elseif base == MathConstants.e
        log
    elseif base == 10
        log10
    else
        x -> log(base, x)
    end
end
