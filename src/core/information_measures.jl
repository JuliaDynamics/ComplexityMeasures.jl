export InformationMeasure
export DiscreteInfoEstimator, DiscreteInfoEstimator
export DifferentialInfoEstimator, DifferentialInfoEstimator
export information, information_maximum, information_normalized, convert_logunit


"""
    InformationMeasure

`InformationMeasure` is the supertype of all information measure definitions.

In this package, we define "information measures" as functionals of probability mass
functions ("discrete" measures), or of probability density functions ("differential"
measures). Examples are (generalized) entropies such as [`Shannon`](@ref) or
[`Renyi`](@ref), or extropies like [`ShannonExtropy`](@ref).
A particular information measure may have both a discrete and a continuous/differential
definition, which are estimated using a [`DifferentialInfoEstimator`](@ref) or
a [`DifferentialInfoEstimator`](@ref), respectively.

## Used with

Any of the information measures listed below can be used with

- [`information`](@ref), to compute a numerical value for the measure, given some input data.
- [`information_maximum`](@ref), to compute the maximum possible value for the measure.
- [`information_normalized`](@ref), to compute the normalized form of the
    measure (divided by the maximum possible value).

The [`information_maximum`](@ref)/[`information_normalized`](@ref) functions only works
with the discrete version of the measure. See docstrings for the above functions
for usage examples.

## Implementations

- [`Renyi`](@ref).
- [`Tsallis`](@ref).
- [`Shannon`](@ref), which is a subcase of the above two in the limit `q → 1`.
- [`Kaniadakis`](@ref).
- [`Curado`](@ref).
- [`StretchedExponential`](@ref).
- [`RenyiExtropy`](@ref).
- [`TsallisExtropy`](@ref).
- [`ShannonExtropy`](@ref), which is a subcase of the above two in the limit `q → 1`.

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

!!! info "Why separate the *definition* of a measure from *estimators* of a measure?"
    In real applications, we generally don't have access to the underlying probability
    mass functions or densities required to compute the various entropy or extropy
    definitons. Therefore, these information measures must be *estimated* from
    finite data. Estimating a particular measure (e.g. [`Shannon`](@ref) entropy) can be
    done in many ways, each with its own own pros and cons. We aim to provide a complete
    library of literature estimators of the various information measures (PRs are welcome!).
"""
abstract type InformationMeasureEstimator{E} end

function information(est::InformationMeasureEstimator, probest::ProbabilitiesEstimator, args...)
    throw(ArgumentError("""$est not implemented for information measure $(est.definition)"""))
end

###########################################################################################
# Discrete entropy
###########################################################################################
"""
    DiscreteInfoEstimator

The supertype of all discrete information measure estimators, which are used in combination
with a [`ProbabilitiesEstimator`](@ref) as input to  [`information`](@ref) or related
functions.

The first argument to a discrete estimator is always an [`InformationMeasure`](@ref)
(defaults to [`Shannon`](@ref)).

## Description

A discrete [`InformationMeasure`](@ref) is a functional of a probability mass function.
To estimate such a measure from data, we must first estimate a probability mass function
using a [`ProbabilitiesEstimator`](@ref) from the (encoded/discretized) input data, and then
apply the estimator to the estimated probabilities. For example, the [`Shannon`](@ref)
entropy is typically computed using the [`RelativeAmount`](@ref) estimator to compute probabilities,
which are then given to the [`PlugIn`](@ref) estimator. Many other estimators exist, not
only for [`Shannon`](@ref) entropy, but other information measures as well.

We provide a library of both generic estimators such as [`PlugIn`](@ref) or
[`Jackknife`](@ref) (which can be applied to any measure), as well as dedicated
estimators such as [`MillerMadow`](@ref), which computes [`Shannon`](@ref) entropy
using the Miller-Madow bias correction. The list below gives a complete overview.

## Implementations

The following estimators are generic and can compute any [`InformationMeasure`](@ref).

- [`PlugIn`](@ref). The default, generic plug-in estimator of any information measure.
    It computes the measure exactly as stated in the definition, using the provided
    probabilities.
- [`Jackknife`](@ref). Uses the a combination of the plug-in estimator and the jackknife
    principle to estimate an [`InformationMeasure`](@ref).

### [`Shannon`](@ref) entropy estimators

The following estimators are dedicated [`Shannon`](@ref) entropy estimators, which
provide improvements over the naive [`PlugIn`](@ref) estimator.

- [`MillerMadow`](@ref).
- [`HorvitzThompson`](@ref).
- [`Schürmann`](@ref).
- [`GeneralizedSchürmann`](@ref).
- [`ChaoShen`](@ref).


!!! info
    Any of the implemented [`DiscreteInfoEstimator`](@ref)s can be used in combination
    with *any* [`ProbabilitiesEstimator`](@ref) as input to [`information`](@ref).
    What this means is that every estimator actually comes in many different variants -
    one for each [`ProbabilitiesEstimator`](@ref). For example, the [`MillerMadow`](@ref)
    estimator of [`Shannon`](@ref) entropy is typically calculated with [`RelativeAmount`](@ref)
    probabilities. But here, you can use for example the [`BayesianRegularization`](@ref) or the
    [`Shrinkage`](@ref) probabilities estimators instead, i.e.
    `information(MillerMadow(), RelativeAmount(outcome_space), x)` and
    `information(MillerMadow(), BayesianRegularization(outcomes_space), x)` are distinct estimators.
    This holds for all [`DiscreteInfoEstimator`](@ref)s. Many of these
    estimators haven't been explored in the literature before, so feel free to explore,
    and please cite this software if you use it to explore some new estimator combination!


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
