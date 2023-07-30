export InformationMeasure
export DiscreteInfoEstimator, DiscreteInfoEstimator
export DifferentialInfoEstimator, DifferentialInfoEstimator
export information, information_maximum, information_normalized, convert_logunit

"""
    InformationMeasure

`InformationMeasure` is the supertype of all information measure definitions, and subtypes
include measures such as (generalized) entropies or extropies.

## Description

Information measures are, in this package, defined as functionals of probability mass
functions ("discrete" measures), or of probability density functions ("differential"
measures).

## Implementations

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

## Estimation

Any of the above measures  may used with [`information`](@ref) to compute a numerical
value, either
- directly (e.g. `information(Shannon(), probest, x)`), which uses [`PlugIn`](@ref)
    estimation,
- with a [`DiscreteInfoEstimator`](@ref) in combination with a [`ProbabilitiesEstimator`](@ref) (e.g. `information(MillerMadow(Shannon()), probest, x)`), or
- with a [`DifferentialInfoEstimator`](@ref) (e.g. `information(Kraskov(Shannon()), x)`).

Why separate the *definition* of a measure from *estimators* of a measure?
In real applications, we generally don't have access to the underlying probability
mass functions or densities. Therefore, information measures must be *estimated* from
finite data. Estimating a particular measure, e.g. [`Shannon`](@ref) entropy can be
done in many ways, each with its own own pros and cons. We aim to provide a complete
library of literature estimators of the various information measures (PRs are welcome!).

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
    throw(ArgumentError("""$est not implemented for information measure $(est.definition)"""))
end

###########################################################################################
# Discrete entropy
###########################################################################################
"""
    DiscreteInfoEstimator

The supertype of all discrete information measure estimators, whose
subtypes of `DiscreteInfoEstimator` are used with [`information`](@ref) to estimate a
discrete [`InformationMeasure`](@ref).

## Implementations

### Generic estimators

- [`PlugIn`](@ref). The default, generic plug-in estimator of any quantity. You can
    actually skip using this estimator, and instead use an [`InformationMeasure`](@ref)
    directly in [`information`](@ref), which will use plug-in estimation by default.
- [`Jackknife`](@ref). Uses the jackknife principle to estimate an
    [`InformationMeasure`](@ref).

### [`Shannon`](@ref) entropy estimators

- [`MillerMadow`](@ref).
- [`HorvitzThompson`](@ref).
- [`Schürmann`](@ref).
- [`GeneralizedSchürmann`](@ref).
- [`ChaoShen`](@ref).

More estimators will be added in the future ([#237](https://github.com/JuliaDynamics/ComplexityMeasures.jl/issues/237)).
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
