export InformationMeasureDefinition
export ML, DiscInfoMeasureEst, DiscreteInformationMeasureEstimator
export DiffInfoMeasureEst, DifferentialInformationMeasureEstimator
export information, information_maximum, information_normalized, convert_logunit

"""
    InformationMeasureDefinition <: ProbabilitiesFunctional

`InformationMeasureDefinition` is the supertype of all types that encapsulate *definitions*
of (generalized) entropies, extropies and other information measures that are defined over
probability distributions or probability densities.

`InformationMeasureDefinition`s also serve directly as naive plug-in *estimators* of
discrete probability functionals; see description below (and [`ML`](@ref)).

## Implementations

Any of the following concrete types can be given as inputs to [`information`](@ref) or
[`information_normalized`](@ref), which will estimate the (normalized) functional.

### Entropies:

- [`Renyi`](@ref).
- [`Tsallis`](@ref).
- [`Shannon`](@ref), which is a subcase of the above two in the limit `q → 1`.
- [`Kaniadakis`](@ref).
- [`Curado`](@ref).
- [`StretchedExponential`](@ref).

### Extropies:

- [`RenyiExtropy`](@ref).
- [`TsallisExtropy`](@ref).
- [`ShannonExtropy`](@ref), which is a subcase of the above two in the limit `q → 1`.

These types can be given as inputs to [`information`](@ref) or [`information_normalized`](@ref).

## Description

Information measures are simply functionals of probability mass functions, or of
probability density functions.

The most commonly used information measures are entropy-based. Mathematically speaking,
generalized entropies are just nonnegative functions of
probability distributions that verify certain (entropy-type-dependent) axioms.
Amigó et al.'s[^Amigó2018] summary paper gives a nice overview.

However, for a software implementation computing entropies _in practice_,
definitions is not really what matters; **estimators matter**.
Because in the practical sense, one needs to estimate some information measure from finite
data, and different ways of estimating a quantity come with their own pros and cons.

Since *estimating* an information is different from the *definition* of an information
measure, we provide the [`DiscreteInformationMeasureEstimator`](@ref) to distinguish
between different estimators of a [`InformationMeasureDefinition`](@ref). And actually,
behind the scenes, it is the *estimator* that is given to [`information`](@ref) when
computing some measure.
Some ways to estimate a discrete information measure only apply to that specific
measure definition. For estimators that can be applied to various measure definitions,
this is specified by providing an instance of [`InformationMeasureDefinition`](@ref) to the
estimator.

[^Amigó2018]:
    Amigó, J. M., Balogh, S. G., & Hernández, S. (2018). A brief review of
    generalized entropies. [Entropy, 20(11), 813.](https://www.mdpi.com/1099-4300/20/11/813)
"""
abstract type InformationMeasureDefinition <: ProbabilitiesFunctional end

###########################################################################################
# Discrete entropy
###########################################################################################
"""
    DiscreteInformationMeasureEstimator
    DiscInfoMeasureEst # alias

Supertype of all discrete information measure estimators.

Currently only the [`ML`](@ref) estimator is provided,
which does not need to be used, as using an [`InformationMeasureDefinition`](@ref) directly in
[`information`](@ref) is possible. But in the future, more advanced estimators will
be added ([#237](https://github.com/JuliaDynamics/ComplexityMeasures.jl/issues/237)).
"""
abstract type DiscreteInformationMeasureEstimator <: ProbabilitiesFunctionalEstimator end
const DiscInfoMeasureEst = DiscreteInformationMeasureEstimator

# Dummy estimator that doesn't actually change anything from the definitions
"""
    ML(e::InformationMeasureDefinition) <: DiscreteInformationMeasureEstimator

The `ML` estimator stands for "maximum likelihood estimator (also called the
empirical/naive/plug-in estimator).

This estimator calculates a quantity exactly as given by some formula, using plug-in
estimates (i.e. observed frequencies). For information measures, for example, it
estimates the [`InformationMeasureDefinition`](@ref) directly from a probability mass
function (which is derived from plug-in estimates of the probabilities).
"""
struct ML{E<:InformationMeasureDefinition} <: DiscreteInformationMeasureEstimator
    definition::E
end

# Notice that StatsBase.jl also exports `information`!
"""
    information([e::DiscreteInformationMeasureEstimator,] probs::Probabilities) → h::Real
    information([e::DiscreteInformationMeasureEstimator,] est::ProbabilitiesEstimator, x) → h::Real

Estimate a **discrete information measure**, using the estimator `est`, in one of two ways:

1. Directly from existing [`Probabilities`](@ref) `probs`.
2. From input data `x`, by first estimating a probability mass function using the provided
   [`ProbabilitiesEstimator`](@ref), and then computing the information measure from that
   mass fuction using the provided [`DiscreteInformationMeasureEstimator`](@ref).

Instead of providing a [`DiscreteInformationMeasureEstimator`](@ref), an
[`InformationMeasureDefinition`](@ref) can be given directly, in which case [`ML`](@ref)
is used as the estimator. If `e` is not provided, [`Shannon`](@ref)`()` is used by default.

## Maximum values and normalized information measures

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
h = information(Shannon(), CountOccurrences(x), x) # syntactically equivalent to above
h = information(SymbolicPermutation(;m=3), x) # gives about 2, again by definition
h = information(Renyi(2.0), ps) # also gives 1, order `q` doesn't matter for coin toss
```
"""
function information(e::InformationMeasureDefinition, est::ProbabilitiesEstimator, x)
    ps = probabilities(est, x)
    return information(e, ps)
end

# dispatch for `information(e::InformationMeasureDefinition, ps::Probabilities)`
# is in the individual entropy definitions files

# Convenience
information(est::ProbabilitiesEstimator, x) = information(Shannon(), est, x)
information(probs::Probabilities) = information(Shannon(), probs)
information(e::ML, args...) = information(e.definition, args...)

###########################################################################################
# Differential entropy
###########################################################################################
"""
    DifferentialInformationMeasureEstimator
    DiffInfoMeasureEst # alias

The supertype of all differential information measure estimators.
These estimators compute an information measure in various ways that do not involve
explicitly estimating a probability distribution.

See the [table of differential information measure estimators](@ref table_diff_ent_est)
in the docs for all differential information measure estimators.

See [`information`](@ref) for usage.
"""
abstract type DifferentialInformationMeasureEstimator <: ProbabilitiesFunctionalEstimator end
const DiffInfoMeasureEst = DifferentialInformationMeasureEstimator

# Dispatch for these functions is implemented in individual estimator files in
# `entropies/estimators/`.
"""
    information(est::DifferentialInformationMeasureEstimator, x) → h::Real

Approximate a **differential information measure** using the provided
[`DifferentialInformationMeasureEstimator`](@ref) and input data `x`.
This method doesn't involve explicitly computing (discretized) probabilities first.

The overwhelming majority of information measure estimators estimate the [`Shannon`](@ref)
entropy. If the same estimator can estimate different information measures (e.g. it can
estimate both [`Shannon`](@ref) and [`Tsallis`](@ref)), then the information measure
is provided as an argument to the estimator itself.

See the [table of differential information measure estimators](@ref table_diff_ent_est)
in the docs for all differential information measure estimators.

## Examples

A standard normal distribution has a base-e differential entropy of `0.5*log(2π) + 0.5`
nats.

```julia
est = Kraskov(k = 5, base = ℯ) # Base `ℯ` for nats.
h = information(est, randn(1_000_000))
abs(h - 0.5*log(2π) - 0.5) # ≈ 0.001
```
"""
function information(::DiffInfoMeasureEst, ::Any) end

information(measure::InformationMeasureDefinition,
    est::DiffInfoMeasureEst, args...) = throw(ArgumentError(
        """`InformationMeasureDefinition` must be given as an argument to `est`
        (if possible), not to `information`.
        """
    ))

information(est::DiffInfoMeasureEst, ::Probabilities) = throw(ArgumentError("""
    InformationMeasureDefinition estimators like $(nameof(typeof(est)))
    are not called with probabilities.
"""))

###########################################################################################
# Normalize API
###########################################################################################
"""
    information_maximum(e::InformationMeasureDefinition, est::ProbabilitiesEstimator, x)

Return the maximum value of a discrete entropy with the given probabilities estimator
and input data `x`. Like in [`outcome_space`](@ref), for some estimators
the concrete outcome space is known without knowledge of input `x`,
in which case the function dispatches to `information_maximum(e, est)`.

    information_maximum(e::InformationMeasureDefinition, L::Int)

Same as above, but computed directly from the number of total outcomes `L`.
"""
function information_maximum(e::InformationMeasureDefinition, est::ProbabilitiesEstimator, x)
    L = total_outcomes(est, x)
    return information_maximum(e, L)
end
function information_maximum(e::InformationMeasureDefinition, est::ProbabilitiesEstimator)
    L = total_outcomes(est)
    return information_maximum(e, L)
end
function information_maximum(e::InformationMeasureDefinition, ::Int)
    error("not implemented for entropy type $(nameof(typeof(e))).")
end
information_maximum(e::ML, args...) = information_maximum(e.definition, args...)

"""
    information_normalized([e::DiscreteInformationMeasureEstimator,] est::ProbabilitiesEstimator, x) → h̃

Return `h̃ ∈ [0, 1]`, the normalized discrete information measure `e` computed from `x`,
i.e. the value of [`information`](@ref) divided by the maximum value for `e`, according to the
given probabilities estimator.

Instead of a discrete information measure estimator, an [`InformationMeasureDefinition`](@ref)
can be given as first argument. If `e` is not given, it defaults to `Shannon()`.

Notice that there is no method
`information_normalized(e::DiscreteInformationMeasureEstimator, probs::Probabilities)`,
because there is no way to know
the amount of _possible_ events (i.e., the [`total_outcomes`](@ref)) from `probs`.
"""
function information_normalized(e::InformationMeasureDefinition, est::ProbabilitiesEstimator, x)
    # If the maximum information is zero (i.e. only one outcome), then we define
    # normalized information as 0.0.
    infomax = information_maximum(e, est, x)
    if infomax == 0
        return 0.0
    end
    return information(e, est, x) / infomax
end
function information_normalized(est::ProbabilitiesEstimator, x::Array_or_SSSet)
    return information_normalized(Shannon(), est, x)
end
information_normalized(e::ML, est, x) = information_normalized(e.definition, est, x)

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

"""
    convert_logunit(h_a::Real, a, b) → h_b

Convert a number `h_a` computed with logarithms to base `a` to an entropy `h_b` computed
with logarithms to base `b`. This can be used to convert the "unit" of an entropy.
"""
function convert_logunit(h_a::Real, base_from, base_to)
    h_a / log(base_from, base_to)
end
