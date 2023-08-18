# [Information measures](@id information_measures)

!!! note
    Please be sure you have read the [Terminology](@ref terminology) section before going through the API here.

## Information measures API

The information measure API is defined by the [`information`](@ref) function, which takes
as an input an [`InformationMeasure`](@ref), or some specialized [`DiscreteInfoEstimator`](@ref) or [`DifferentialInfoEstimator`](@ref) for estimating the discrete or differential variant
of the measure.

The functions [`information_maximum`](@ref) and [`information_normalized`](@ref) are also useful.

## Definitions (entropies and extropies)

```@docs
InformationMeasure
Shannon
Renyi
Tsallis
Kaniadakis
Curado
StretchedExponential
ShannonExtropy
RenyiExtropy
TsallisExtropy
```

## Estimation (discrete)

```@docs
information(::InformationMeasure, ::OutcomeSpace, ::Any)
information_maximum
information_normalized
```

### Discrete information estimators

```@docs
DiscreteInfoEstimator
PlugIn
MillerMadow
Schürmann
GeneralizedSchürmann
Jackknife
HorvitzThompson
ChaoShen
```

## Estimation (differential)

```@docs
information(::DifferentialInfoEstimator, ::Any)
```

### Differential information estimators

```@docs
DifferentialInfoEstimator
```

```@docs
Kraskov
KozachenkoLeonenko
Zhu
ZhuSingh
Gao
Goria
Lord
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
