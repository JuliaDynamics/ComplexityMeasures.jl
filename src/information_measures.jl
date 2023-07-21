export InformationMeasure
export PlugIn, DiscreteInfoEstimator, DiscreteInfoEstimator
export DifferentialInfoEstimator, DifferentialInfoEstimator
export information, information_maximum, information_normalized, convert_logunit

"""
    InformationMeasure

`InformationMeasure` is the supertype of all information measures, including measures
such as (generalized) entropies or extropies.

Within ComplexityMeasures.jl, we have taken the pragmatic choice to label all
measures that are **functionals of probability mass or density functions**
as **information measures**, even though they might not be labelled as
information measures in the literature.

## Definitions

Any of the following definitions appear in either discrete form, differential form, or
as both. They can be given as input at least one of the implemented
[`DiscreteInfoEstimator`](@ref)s or
[`DifferentialInfoEstimator`](@ref)s. In turn, the estimator is given as
input to [`information`](@ref) or [`information_normalized`](@ref) to compute the numeric
value corresponding to the measure.

### Entropies

- [`Renyi`](@ref).
- [`Tsallis`](@ref).
- [`Shannon`](@ref), which is a subcase of the above two in the limit `q → 1`.
- [`Kaniadakis`](@ref).
- [`Curado`](@ref).
- [`StretchedExponential`](@ref).

### Extropies

- [`RenyiExtropy`](@ref).
- [`TsallisExtropy`](@ref).
- [`ShannonExtropy`](@ref), which is a subcase of the above two in the limit `q → 1`.

## Description

Information measures are simply functionals of probability mass functions, or of
probability density functions. The most commonly used information measures are
(generalized) entropies, which are just nonnegative functions of probability distributions
that verify certain (entropy-type-dependent) axioms. Amigó et al.'s[^Amigó2018] summary
paper gives a nice overview.

However, in practice, the situation is a bit more complicated, because *estimating* an
information measure is different from the *definition* of an information measure. A
measure must be estimated from finite data, and there are many different ways
of estimating a given quantity, each with its own own pros and cons.
[`DiscreteInfoEstimator`](@ref)s and [`DifferentialInfoEstimator`](@ref)s are provided
to distinguish between different estimators of a discrete and differential variants of an
[`InformationMeasure`](@ref), respectively.

[^Amigó2018]:
    Amigó, J. M., Balogh, S. G., & Hernández, S. (2018). A brief review of
    generalized entropies. [Entropy, 20(11), 813.](https://www.mdpi.com/1099-4300/20/11/813)
"""
abstract type InformationMeasure end

"""
    InformationMeasureEstimator{I <: InformationMeasure}

The supertype of all information measure estimators. Any concrete subtypes must be
parametric types where the first type is some [`InformationMeasure`](@ref),
i.e.

```julia
struct MyEstimator{I <: InformationMeasure} <: InformationMeasureEstimator{I}
    e::I
    params...
end
```

We use [`DifferentialInfoEstimator`](@ref) for estimating differential quantities,
and [`DiscreteInfoEstimator`](@ref) for estimating discrete quantities.
"""
abstract type InformationMeasureEstimator{E} end

function information(est::InformationMeasureEstimator, args...)
    throw(ArgumentError("""$est not implemented for information measure $(est.measure)"""))
end

###########################################################################################
# Discrete entropy
###########################################################################################
"""
    DiscreteInfoEstimator

Supertype of all discrete information measure estimators.

## Implementations

Currently only the [`PlugIn`](@ref) estimator is provided,
which does not need to be used, as using an [`InformationMeasure`](@ref) directly in
[`information`](@ref) is possible. But in the future, more advanced estimators will
be added ([#237](https://github.com/JuliaDynamics/ComplexityMeasures.jl/issues/237)).
"""
abstract type DiscreteInfoEstimator{I <: InformationMeasure} <: InformationMeasureEstimator{I} end

# Dummy estimator that doesn't actually change anything from the definitions
"""
    PlugIn(e::InformationMeasure) <: DiscreteInfoEstimator

The `PlugIn` estimator is also called the empirical/naive/"maximum likelihood" estimator.

This estimator calculates a quantity exactly as given by its formula, using plug-in
estimates (i.e. observed frequencies). For information measures, for example, it
estimates the [`InformationMeasure`](@ref) directly from a probability mass
function (which is derived from plug-in estimates of the probabilities).
"""
struct PlugIn{I <: InformationMeasure} <: DiscreteInfoEstimator{I}
    definition::I
end

# Notice that StatsBase.jl also exports `information`!
"""
    information([e::DiscreteInfoEstimator,] probs::Probabilities) → h::Real
    information([e::DiscreteInfoEstimator,] est::ProbabilitiesEstimator, x) → h::Real

Estimate a **discrete information measure**, using the estimator `est`, in one of two ways:

1. Directly from existing [`Probabilities`](@ref) `probs`.
2. From input data `x`, by first estimating a probability mass function using the provided
   [`ProbabilitiesEstimator`](@ref), and then computing the information measure from that
   mass fuction using the provided [`DiscreteInfoEstimator`](@ref).

Instead of providing a [`DiscreteInfoEstimator`](@ref), an
[`InformationMeasure`](@ref) can be given directly, in which case [`PlugIn`](@ref)
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
h = information(PlugIn(Shannon()), ps) # syntactically equivalent to the above
h = information(PlugIn(Shannon()), CountOccurrences(x), x) # syntactically equivalent to above
h = information(PlugIn(SymbolicPermutation(;m=3)), x) # gives about 2, again by definition
h = information(PlugIn(Renyi(2.0)), ps) # also gives 1, order `q` doesn't matter for coin toss
```
"""
function information(e::InformationMeasure, est::ProbabilitiesEstimator, x)
    ps = probabilities(est, x)
    return information(e, ps)
end

# dispatch for `information(e::InformationMeasure, ps::Probabilities)`
# is in the individual entropy definitions files

# Convenience
information(est::ProbabilitiesEstimator, x) = information(Shannon(), est, x)
information(probs::Probabilities) = information(Shannon(), probs)
information(e::PlugIn, args...) = information(e.definition, args...)

###########################################################################################
# Differential entropy
###########################################################################################
"""
    DifferentialInfoEstimator

The supertype of all differential information measure estimators.
These estimators compute an information measure in various ways that do not involve
explicitly estimating a probability distribution.

Each [`DifferentialInfoEstimator`](@ref)s uses a specialized technique to approximate
relevant densities/integrals, and is often tailored to one or a few types of information
measures. For example, [`Kraskov`](@ref) estimates the [`Shannon`](@ref) entropy.

See [`information`](@ref) for usage.

## Implementations

- [`KozachenkoLeonenko`](@ref).
- [`Kraskov`](@ref).
- [`Goria`](@ref).
- [`Gao`](@ref).
- [`Zhu`](@ref)
- [`ZhuSingh`](@ref).
- [`Lord`](@ref).
- [`AlizadehArghami`](@ref).
- [`Correa`](@ref).
- [`Vasicek`](@ref).
- [`Ebrahimi`](@ref).
"""
abstract type DifferentialInfoEstimator{I <: InformationMeasure} <: InformationMeasureEstimator{I} end

# Dispatch for these functions is implemented in individual estimator files in
# `entropies/estimators/`.
"""
    information(est::DifferentialInfoEstimator, x) → h::Real

Approximate a **differential information measure** using the provided
[`DifferentialInfoEstimator`](@ref) and input data `x`.
This method doesn't involve explicitly computing (discretized) probabilities first.

The overwhelming majority of information measure estimators estimate the [`Shannon`](@ref)
entropy. If the same estimator can estimate different information measures (e.g. it can
estimate both [`Shannon`](@ref) and [`Tsallis`](@ref)), then the information measure
is provided as an argument to the estimator itself.

See the [table of differential information measure estimators](@ref table_diff_ent_est)
in the docs for all differential information measure estimators.

## Examples

A standard normal distribution has a base-e Shannon differential entropy of
`0.5*log(2π) + 0.5` nats.

```julia
est = Kraskov(measure = Shannon(), k = 5, base = ℯ) # Base `ℯ` for nats.
h = information(est, randn(2_000_000))
abs(h - 0.5*log(2π) - 0.5) # ≈ 0.0001
```
"""
function information(est::DifferentialInfoEstimator{<:I}, args...) where I
    throw(ArgumentError("""$est not implemented for information measure of type $I"""))
end

information(measure::InformationMeasure,
    est::DifferentialInfoEstimator, args...) = throw(ArgumentError(
        """`InformationMeasure` must be given as an argument to `est`, not to `information`.
        """
    ))

# TODO: This collides with some other definition. Figure out what.
# information(est::DifferentialInfoEstimator, ::Probabilities) = throw(ArgumentError("""
#     InformationMeasure estimators like $(nameof(typeof(est)))
#     are not called with probabilities.
# """))

###########################################################################################
# Normalize API
###########################################################################################
"""
    information_maximum(e::InformationMeasure, est::ProbabilitiesEstimator, x)

Return the maximum value of a discrete entropy with the given probabilities estimator
and input data `x`. Like in [`outcome_space`](@ref), for some estimators
the concrete outcome space is known without knowledge of input `x`,
in which case the function dispatches to `information_maximum(e, est)`.

    information_maximum(e::InformationMeasure, L::Int)

Same as above, but computed directly from the number of total outcomes `L`.
"""
function information_maximum(e::InformationMeasure, est::ProbabilitiesEstimator, x)
    L = total_outcomes(est, x)
    return information_maximum(e, L)
end
function information_maximum(e::InformationMeasure, est::ProbabilitiesEstimator)
    L = total_outcomes(est)
    return information_maximum(e, L)
end
function information_maximum(e::InformationMeasure, ::Int)
    throw(ErrorException("not implemented for entropy type $(nameof(typeof(e)))."))
end
information_maximum(e::PlugIn, args...) = information_maximum(e.definition, args...)

"""
    information_normalized([e::DiscreteInfoEstimator,] est::ProbabilitiesEstimator, x) → h̃

Return `h̃ ∈ [0, 1]`, the normalized discrete information measure `e` computed from `x`,
i.e. the value of [`information`](@ref) divided by the maximum value for `e`, according to the
given probabilities estimator.

Instead of a discrete information measure estimator, an [`InformationMeasure`](@ref)
can be given as first argument. If `e` is not given, it defaults to `Shannon()`.

Notice that there is no method
`information_normalized(e::DiscreteInfoEstimator, probs::Probabilities)`,
because there is no way to know
the amount of _possible_ events (i.e., the [`total_outcomes`](@ref)) from `probs`.
"""
function information_normalized(e::InformationMeasure, est::ProbabilitiesEstimator, x)
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
information_normalized(e::PlugIn, est, x) = information_normalized(e.definition, est, x)

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
