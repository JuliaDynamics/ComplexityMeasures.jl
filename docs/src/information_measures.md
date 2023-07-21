
# [Information measures](@id information_measures)

Please be sure you have read the [Terminology](@ref terminology) section before going through the API here, to have a good idea of the different "flavors" of information measures and how they all come together over the common interface of the [`information`](@ref) function.

## Information measures API

The information measure API is defined by

- [`InformationMeasure`](@ref)
- [`information`](@ref)
- [`information_maximum`](@ref)
- [`information_normalized`](@ref)
- [`InformationMeasureEstimator`](@ref)
- [`DiscreteInfoEstimator`](@ref)
- [`DifferentialInfoEstimator`](@ref)

## Information measures definitions

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

## Discrete information measures

```@docs
information(::InformationMeasure, ::ProbabilitiesEstimator, ::Any)
information_maximum
information_normalized
```

### Discrete information measure estimators

```@docs
DiscreteInfoEstimator
PlugIn
```

## Differential information measure

```@docs
information(::DifferentialInfoEstimator, ::Any)
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