export EntropyDefinition
export MLEntropy, DiscEntropyEst, DiscreteEntropyEstimator
export DiffEntropyEst, DifferentialEntropyEstimator
export entropy, entropy_maximum, entropy_normalized

"""
    EntropyDefinition

`EntropyDefinition` is the supertype of all types that encapsulate definitions
of (generalized) entropies. These also serve as estimators of discrete entropies,
see description below.

Currently implemented entropy definitions are:

- [`Renyi`](@ref).
- [`Tsallis`](@ref).
- [`Shannon`](@ref), which is a subcase of the above two in the limit `q → 1`.
- [`Kaniadakis`](@ref).
- [`Curado`](@ref).
- [`StretchedExponential`](@ref).

These types can be given as inputs to [`entropy`](@ref) or [`entropy_normalized`](@ref).

## Description

Mathematically speaking, generalized entropies are just nonnegative functions of
probability distributions that verify certain (entropy-type-dependent) axioms.
Amigó et al.'s[^Amigó2018] summary paper gives a nice overview.

However, for a software implementation computing entropies _in practice_,
definitions is not really what matters; **estimators matter**.
Because in the practical sense, one needs to estimate a definition from finite data,
and different ways of estimating a quantity come with their own pros and cons.

That is why the type [`DiscreteEntropyEstimator`](@ref) exists,
which is what is actually given to [`entropy`](@ref).
Some ways to estimate a discrete entropy only apply to a specific entropy definition.
For estimators that can be applied to various entropy definitions,
this is specified by providing an instance of `EntropyDefinition` to the estimator.

[^Amigó2018]:
    Amigó, J. M., Balogh, S. G., & Hernández, S. (2018). A brief review of
    generalized entropies. [Entropy, 20(11), 813.](https://www.mdpi.com/1099-4300/20/11/813)
"""
abstract type EntropyDefinition end

###########################################################################################
# Discrete entropy
###########################################################################################
"""
    DiscreteEntropyEstimator
    DiscEntropyEst # alias

Supertype of all discrete entropy estimators.

Currently only the [`MLEntropy`](@ref) estimator is provided,
which does not need to be used, as using an [`EntropyDefinition`](@ref) directly in
[`entropy`](@ref) is possible. But in the future, more advanced estimators will
be added ([#237](https://github.com/JuliaDynamics/ComplexityMeasures.jl/issues/237)).
"""
abstract type DiscreteEntropyEstimator end
const DiscEntropyEst = DiscreteEntropyEstimator

# Dummy estimator that doesn't actually change anything from the definitions
"""
    MLEntropy(e::EntropyDefinition) <: DiscreteEntropyEstimator

Standing for "maximum likelihood entropy", and also called empirical/naive/plug-in,
this estimator calculates the entropy exactly as defined in the given
[`EntropyDefinition`](@ref) directly from a probability mass function.
"""
struct MLEntropy{E<:EntropyDefinition} <: DiscreteEntropyEstimator
    definition::E
end

# Notice that StatsBase.jl also exports `entropy`!
"""
    entropy([e::DiscreteEntropyEstimator,] probs::Probabilities)
    entropy([e::DiscreteEntropyEstimator,] est::ProbabilitiesEstimator, x)

Compute the **discrete entropy** `h::Real ∈ [0, ∞)`,
using the estimator `e`, in one of two ways:

1. Directly from existing [`Probabilities`](@ref) `probs`.
2. From input data `x`, by first estimating a probability mass function using the provided
   [`ProbabilitiesEstimator`](@ref), and then computing the entropy from that mass fuction
   using the provided [`DiscreteEntropyEstimator`](@ref).

Instead of providing a [`DiscreteEntropyEstimator`](@ref), an [`EntropyDefinition`](@ref)
can be given directly, in which case [`MLEntropy`](@ref) is used as the estimator.
If `e` is not provided, [`Shannon`](@ref)`()` is used by default.

## Maximum entropy and normalized entropy

All discrete entropies have a well defined maximum value for a given probability estimator.
To obtain this value one only needs to call the [`entropy_maximum`](@ref).
Or, one can use [`entropy_normalized`](@ref)
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

# dispatch for `entropy(e::EntropyDefinition, ps::Probabilities)`
# is in the individual entropy definitions files

# Convenience
entropy(est::ProbabilitiesEstimator, x) = entropy(Shannon(), est, x)
entropy(probs::Probabilities) = entropy(Shannon(), probs)
entropy(e::MLEntropy, args...) = entropy(e.definition, args...)

###########################################################################################
# Differential entropy
###########################################################################################
"""
    DifferentialEntropyEstimator
    DiffEntropyEst # alias

The supertype of all differential entropy estimators.
These estimators compute an entropy value in various ways that do not involve
explicitly estimating a probability distribution.

See the [table of differential entropy estimators](@ref table_diff_ent_est)
in the docs for all differential entropy estimators.

See [`entropy`](@ref) for usage.
"""
abstract type DifferentialEntropyEstimator end
const DiffEntropyEst = DifferentialEntropyEstimator

# Dispatch for these functions is implemented in individual estimator files in
# `entropies/estimators/`.
"""
    entropy(est::DifferentialEntropyEstimator, x) → h

Approximate the **differential entropy** `h::Real` using the provided
[`DifferentialEntropyEstimator`](@ref) and input data `x`.
This method doesn't involve explicitly computing (discretized) probabilities first.

The overwhelming majority of entropy estimators estimate the Shannon entropy.
If some estimator can estimate different _definitions_ of entropy (e.g., [`Tsallis`](@ref)),
this is provided as an argument to the estimator itself.

See the [table of differential entropy estimators](@ref table_diff_ent_est)
in the docs for all differential entropy estimators.

## Examples

A standard normal distribution has a base-e differential entropy of `0.5*log(2π) + 0.5`
nats.

```julia
est = Kraskov(k = 5, base = ℯ) # Base `ℯ` for nats.
h = entropy(est, randn(1_000_000))
abs(h - 0.5*log(2π) - 0.5) # ≈ 0.001
```
"""
function entropy(::DiffEntropyEst, ::Any) end

entropy(est::DiffEntropyEst, ::Probabilities) = error("""
    EntropyDefinition estimators like $(nameof(typeof(est)))
    are not called with probabilities.
""")

###########################################################################################
# Normalize API
###########################################################################################
"""
    entropy_maximum(e::EntropyDefinition, est::ProbabilitiesEstimator, x)

Return the maximum value of a discrete entropy with the given probabilities estimator
and input data `x`. Like in [`outcome_space`](@ref), for some estimators
the concrete outcome space is known without knowledge of input `x`,
in which case the function dispatches to `entropy_maximum(e, est)`.

    entropy_maximum(e::EntropyDefinition, L::Int)

Same as above, but computed directly from the number of total outcomes `L`.
"""
function entropy_maximum(e::EntropyDefinition, est::ProbabilitiesEstimator, x)
    L = total_outcomes(est, x)
    return entropy_maximum(e, L)
end
function entropy_maximum(e::EntropyDefinition, est::ProbabilitiesEstimator)
    L = total_outcomes(est)
    return entropy_maximum(e, L)
end
function entropy_maximum(e::EntropyDefinition, ::Int)
    error("not implemented for entropy type $(nameof(typeof(e))).")
end
entropy_maximum(e::MLEntropy, args...) = entropy_maximum(e.definition, args...)

"""
    entropy_normalized([e::DiscreteEntropyEstimator,] est::ProbabilitiesEstimator, x) → h̃

Return `h̃ ∈ [0, 1]`, the normalized discrete entropy of `x`, i.e. the value of [`entropy`](@ref)
divided by the maximum value for `e`, according to the given probabilities estimator.

Instead of a discrete entropy estimator, an [`EntropyDefinition`](@ref)
can be given as first argument. If `e` is not given, it defaults to `Shannon()`.

Notice that there is no method
`entropy_normalized(e::DiscreteEntropyEstimator, probs::Probabilities)`,
because there is no way to know
the amount of _possible_ events (i.e., the [`total_outcomes`](@ref)) from `probs`.
"""
function entropy_normalized(e::EntropyDefinition, est::ProbabilitiesEstimator, x)
    return entropy(e, est, x) / entropy_maximum(e, est, x)
end
function entropy_normalized(est::ProbabilitiesEstimator, x::Array_or_Dataset)
    return entropy_normalized(Shannon(), est, x)
end
entropy_normalized(e::MLEntropy, est, x) = entropy_normalized(e.definition, est, x)

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
