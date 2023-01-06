export EntropyDefinition, DiffEntropyEst, DifferentialEntropyEstimator
export entropy, entropy_maximum, entropy_normalized

"""
    EntropyDefinition

`EntropyDefinition` is the supertype of all types that encapsulate definitions
of (generalized) entropies. Currently implemented entropy definitions are:

- [`Renyi`](@ref).
- [`Tsallis`](@ref).
- [`Shannon`](@ref), which is a subcase of the above two in the limit `q → 1`.
- [`Kaniadakis`](@ref).
- [`Curado`](@ref).
- [`StretchedExponential`](@ref).

These entropy types are given as inputs to [`entropy`](@ref) and
[`entropy_normalized`](@ref). Notice that in all documentation strings formulas are
provided for the discrete version of the entropy, for simplicity.

See [`entropy`](@ref) for usage.

## Description

Mathematically speaking, generalized entropies are just nonnegative functions of
probability distributions that verify certain (entropy-type-dependent) axioms.
Amigó et al.'s[^Amigó2018] summary paper gives a nice overview.

[^Amigó2018]:
    Amigó, J. M., Balogh, S. G., & Hernández, S. (2018). A brief review of
    generalized entropies. [Entropy, 20(11), 813.](https://www.mdpi.com/1099-4300/20/11/813)
"""
abstract type EntropyDefinition end

"""
    DifferentialEntropyEstimator
    DiffEntropyEst # alias

The supertype of all differential entropy estimators.
These estimators compute an entropy value in various ways that do not involve
explicitly estimating a probability distribution.

Currently implemented estimators are:

- [`KozachenkoLeonenko`](@ref)
- [`Kraskov`](@ref)
- [`Zhu`](@ref)
- [`ZhuSingh`](@ref)
- [`Goria`](@ref)
- [`Gao`](@ref)
- [`Lord`](@ref)
- [`Vasicek`](@ref)
- [`Ebrahimi`](@ref)
- [`Correa`](@ref)
- [`AlizadehArghami`](@ref)

See [`entropy`](@ref) for usage.
"""
abstract type DifferentialEntropyEstimator end
const DiffEntropyEst = DifferentialEntropyEstimator

###########################################################################################
# Discrete entropy
###########################################################################################
# Notice that StatsBase.jl also exports `entropy`.
"""
    entropy([e::EntropyDefinition,] probs::Probabilities)
    entropy([e::EntropyDefinition,] est::ProbabilitiesEstimator, x)

Compute the **discrete entropy** `h::Real ∈ [0, ∞)` defined by `e`, in one of two ways:

1. Directly from existing [`Probabilities`](@ref) `probs`.
2. From input data `x`, by first estimating a probability distribution using the provided
   [`ProbabilitiesEstimator`](@ref), then computing entropy from that distribution.
   In fact, the second method is just a 2-lines-of-code wrapper that calls
   [`probabilities`](@ref) and gives the result to the first method.

The entropy definition (first argument) is optional. Explicitly provide `e` if you need to
specify a logarithm base for the entropy. When `est` is a probability estimator,
`Shannon()` is used by default.

## Maximum entropy and normalized entropy

All discrete entropies `e` have a well defined maximum value for a given probability estimator.
To obtain this value one only needs to call the [`entropy_maximum`](@ref) function with the
chosen entropy type and probability estimator. Or, one can use [`entropy_normalized`](@ref)
to obtain the normalized form of the entropy (divided by the maximum).

## Examples

```julia
x = [rand(Bool) for _ in 1:10000] # coin toss
ps = probabilities(x) # gives about [0.5, 0.5] by definition
h = entropy(ps) # gives 1, about 1 bit by definition
h = entropy(Shannon(), ps) # syntactically equivalent to above
h = entropy(Shannon(), CountOccurrences(x), x) # syntactically equivalent to above
h = entropy(SymbolicPermutation(;m=3), x) # gives about 2, again by definition
h = entropy(Renyi(2.0), ps) # also gives 1, order `q` doesn't matter for coin toss
```
"""
function entropy(e::EntropyDefinition, est::ProbabilitiesEstimator, x)
    ps = probabilities(est, x)
    return entropy(e, ps)
end

# dispatch for `entropy(e, ps)` in the entropy definitions files

# Convenience
entropy(est::ProbabilitiesEstimator, x) = entropy(Shannon(), est, x)
entropy(probs::Probabilities) = entropy(Shannon(), probs)

###########################################################################################
# Differential entropy
###########################################################################################
# Dispatch for these functions is implemented in individual estimator files in
# `entropies/estimators/`.
"""
    entropy(est::DifferentialEntropyEstimator, x)

Approximate the **differential entropy** `h::Real` using the provided
[`DifferentialEntropyEstimator`](@ref) and input data `x`.
This method doesn't involve explicitly computing (discretized) probabilities first.

The overwhelming majority of entropy estimators estimate the Shannon entropy.
If some estimator can estimate different _definitions_ of entropy (e.g., [`Tsallis`](@ref)),
this is provided as an argument to the estimator itself.

See the [Table of differential entropy estimators](@ref)
in the docs for a table view of all differential entropy estimators.

## Examples

A standard normal distribution has a base-e differential entropy of `0.5*log(2π) + 0.5`
nats.

```julia
est = Kraskov(k = 5, base = ℯ) # Base `ℯ` for nats.
h = entropy(est, randn(1_000_000))
abs(h - 0.5*log(2π) - 0.5) # ≈ 0.001
```
"""
function entropy(e::EntropyDefinition, est::DiffEntropyEst, x)
    entropy_definition_compatibility(e, est)
    return entropy(est, x)
end
entropy(est::DiffEntropyEst, ::Probabilities) = error("""
    EntropyDefinition estimators like $(nameof(typeof(est)))
    are not called with probabilities.
""")

###########################################################################################
# Normalize API
###########################################################################################
"""
    entropy_maximum(e::EntropyDefinition, est::ProbabilitiesEstimator)

Return the maximum value of a discrete entropy the given probabilities estimator.

    entropy_maximum(e::EntropyDefinition, L::Int)

Same as above, but computed thdirectly from the number of total outcomes `L`.
"""
function entropy_maximum(e::EntropyDefinition, est::ProbabilitiesEstimator)
    L = total_outcomes(est)
    return entropy_maximum(e, L)
end
function entropy_maximum(e::EntropyDefinition, ::Int)
    error("not implemented for entropy type $(nameof(typeof(e))).")
end

"""
    entropy_normalized([e::EntropyDefinition,] est::ProbabilitiesEstimator, x) → h̃

Return `h̃ ∈ [0, 1]`, the normalized discrete entropy of `x`, i.e. the value of [`entropy`](@ref)
divided by the maximum value for `e`, according to the given probabilities estimator.
If `e` is not given, it defaults to `Shannon()`.

Notice that there is no method
`entropy_normalized(e::EntropyDefinition, probs::Probabilities)`, because there is no way to know
the amount of _possible_ events (i.e., the [`total_outcomes`](@ref)) from `probs`.
"""
function entropy_normalized(e::EntropyDefinition, est::ProbabilitiesEstimator, x)
    return entropy(e, est, x) / entropy_maximum(e, est)
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
