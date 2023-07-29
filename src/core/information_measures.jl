export InformationMeasure
export DiscreteInfoEstimator, DiscreteInfoEstimator
export DifferentialInfoEstimator, DifferentialInfoEstimator
export information, information_maximum, information_normalized, convert_logunit

"""
    InformationMeasure

`InformationMeasure` is the supertype of all information measures, including measures
such as (generalized) entropies or extropies.

## Existing measures

Any of the following **measure definitions** may appear in either discrete form,
differential form, or as both. They can be given as input at least one of the implemented
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

*Estimating* an information measure may be different from its *definition* due to
finite data, and there could be are many different ways
of estimating a measure, each with its own own pros and cons.
[`DiscreteInfoEstimator`](@ref)s and [`DifferentialInfoEstimator`](@ref)s are provided
to distinguish between different estimators of a discrete and differential variants of an
[`InformationMeasure`](@ref), respectively.

[^Amigó2018]:
    Amigó, J. M., Balogh, S. G., & Hernández, S. (2018). A brief review of
    generalized entropies. [Entropy, 20(11), 813.](https://www.mdpi.com/1099-4300/20/11/813)
"""
abstract type InformationMeasure end

# This is internal: not exported or in public API
"""
    InformationMeasureEstimator{I <: InformationMeasure}

The supertype of all information measure estimators. Its direct subtypes
are [`DiscreteInfoEstimator`](@ref) and [`DifferentialInfoEstimator`](@ref).

Since all estimators must reference a measure definition in some way,
we made the following interface decisions:

1. all estimators have as first type parameter `I <: InformationMeasure`
2. all estimators reference the information measure in a `definition` field
3. all estimators are defined using `Base.@kwdef` so that they may be initialized
   with the syntax `Estimator(; definition = Shannon())` (or any other).

Any concrete subtypes must follow the above, e.g.:

```julia
Base.@kwdef struct MyEstimator{I <: InformationMeasure, X} <: DiscreteInfoEstimator{I}
    definition::I
    x::X
end
```
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
