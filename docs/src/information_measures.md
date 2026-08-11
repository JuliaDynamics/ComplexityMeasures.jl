# [Information measures (entropies and co.)](@id information_measures)

!!! note
    Be sure you have gone through the [Tutorial](@ref) before going through the API here to have a good idea of the terminology used in ComplexityMeasures.jl.


## Information measures API

The information measure API is defined by the [`information`](@ref) function, which takes as an input an [`InformationMeasure`](@ref), or some specialized [`DiscreteInfoEstimator`](@ref) or [`DifferentialInfoEstimator`](@ref) for estimating the discrete or differential variant of the measure.
The functions [`information_maximum`](@ref) and [`information_normalized`](@ref) are also useful.

```@docs
InformationMeasure
information(::InformationMeasure, ::OutcomeSpace, ::Any)
information(::DifferentialInfoEstimator, ::Any)
information_maximum
information_normalized
self_information
```

## [Units and the `base` keyword](@id units_and_base)

`base` sets the scale of a measure, never its shape: two bases differ only by the constant
factor ``1/\ln{(base)}``. For [`Shannon`](@ref) and the other logarithmic measures this is
the familiar choice of unit — `base = 2` gives bits, `base = MathConstants.e` gives nats.

For [`Tsallis`](@ref), [`Kaniadakis`](@ref), [`TsallisExtropy`](@ref) and
[`StretchedExponential`](@ref) the word "unit" is looser, because their defining
expressions contain no logarithm to take the base of; at, say, `q = 1.5` the value is
simply a dimensionless number. What fixes the scale instead is a limit. Each of these
measures is built from a deformed logarithm that becomes an ordinary logarithm at one
parameter value (``q \to 1``, ``\kappa \to 0``, ``\eta = 1``), and their source papers
normalize that limit to the *natural* logarithm. `base = MathConstants.e` therefore
reproduces the published expressions exactly, while any other `base` rescales them
uniformly so that the limiting logarithm is to that base instead. The scaling is applied
at every parameter value, so these limits are continuous and agree exactly with
[`Shannon`](@ref) at the same base.

[`Curado`](@ref) and [`Identification`](@ref) take no `base` at all: they are algebraic
rather than logarithmic and have no such limit, so no scale convention applies to them.

## Entropies

```@docs
entropy
Shannon
Renyi
Tsallis
Kaniadakis
Curado
StretchedExponential
```

## Other information measures

```@docs
ShannonExtropy
RenyiExtropy
TsallisExtropy
ElectronicEntropy
InformationFluctuation
```

## Discrete information estimators

```@docs
DiscreteInfoEstimator
PlugIn
MillerMadow
Schuermann
GeneralizedSchuermann
Jackknife
HorvitzThompson
ChaoShen
```

## Differential information estimators

```@docs
DifferentialInfoEstimator
Kraskov
KozachenkoLeonenko
Zhu
ZhuSingh
Gao
Goria
Lord
LeonenkoProzantoSavani
Vasicek
AlizadehArghami
Ebrahimi
Correa
```

### [Table of differential information measure estimators](@id table_diff_ent_est)

The following estimators are *differential* information measure estimators, and can also be used
with [`information`](@ref).

Each [`DifferentialInfoEstimator`](@ref)s uses a specialized technique to approximate relevant
densities/integrals, and is often tailored to one or a few types of information measures.
For example, [`Kraskov`](@ref) estimates the [`Shannon`](@ref) entropy.

| Estimator                    | Principle         | Input data | [`Shannon`](@ref) | [`Renyi`](@ref) | [`Tsallis`](@ref) | [`Kaniadakis`](@ref) | [`Curado`](@ref) | [`StretchedExponential`](@ref) |
| :--------------------------- | :---------------- | :--------- | :---------------: | :-------------: | :---------------: | :------------------: | :--------------: | :----------------------------: |
| [`KozachenkoLeonenko`](@ref) | Nearest neighbors | `StateSpaceSet`  |        ✓         |        x        |         x         |          x           |        x         |               x                |
| [`Kraskov`](@ref)            | Nearest neighbors | `StateSpaceSet`  |        ✓         |        x        |         x         |          x           |        x         |               x                |
| [`Zhu`](@ref)                | Nearest neighbors | `StateSpaceSet`  |        ✓         |        x        |         x         |          x           |        x         |               x                |
| [`ZhuSingh`](@ref)           | Nearest neighbors | `StateSpaceSet`  |        ✓         |        x        |         x         |          x           |        x         |               x                |
| [`Gao`](@ref)                | Nearest neighbors | `StateSpaceSet`  |        ✓         |        x        |         x         |          x           |        x         |               x                |
| [`Goria`](@ref)              | Nearest neighbors | `StateSpaceSet`  |        ✓         |        x        |         x         |          x           |        x         |               x                |
| [`Lord`](@ref)               | Nearest neighbors | `StateSpaceSet`  |        ✓         |        x        |         x         |          x           |        x         |               x                |
| [`Vasicek`](@ref)            | Order statistics  | `Vector`   |        ✓         |        x        |         x         |          x           |        x         |               x                |
| [`Ebrahimi`](@ref)           | Order statistics  | `Vector`   |        ✓         |        x        |         x         |          x           |        x         |               x                |
| [`Correa`](@ref)             | Order statistics  | `Vector`   |        ✓         |        x        |         x         |          x           |        x         |               x                |
| [`AlizadehArghami`](@ref)    | Order statistics  | `Vector`   |        ✓         |        x        |         x         |          x           |        x         |               x                |
