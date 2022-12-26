# [Entropies](@id entropies)

TODO: Guiding statements here.

## Entropy definitions

```@docs
EntropyDefinition
Shannon
Renyi
Tsallis
Kaniadakis
Curado
StretchedExponential
```

## Discrete entropy

```@docs
entropy(::EntropyDefinition, ::ProbabilitiesEstimator, ::Any)
```

## Differential entropy

```@docs
entropy(::EntropyDefinition, ::DiffEntropyEst, x)
```

### Table of differential entropy estimators

The following estimators are *differential* entropy estimators, and can also be used
with [`entropy`](@ref). Differential) entropies are functions of *integrals*, and usually
rely on estimating some density functional.

Each [`DiffEntropyEst`](@ref)s uses a specialized technique to approximating relevant
densities/integrals, and is often tailored to one or a few types of generalized entropy.
For example, [`Kraskov`](@ref) estimates the [`Shannon`](@ref) entropy.

| Estimator                    | Principle         | Input data | [`Shannon`](@ref) | [`Renyi`](@ref) | [`Tsallis`](@ref) | [`Kaniadakis`](@ref) | [`Curado`](@ref) | [`StretchedExponential`](@ref) |
| ---------------------------- | ----------------- | ---------- | :---------------: | :-------------: | :---------------: | :------------------: | :--------------: | :----------------------------: |
| [`KozachenkoLeonenko`](@ref) | Nearest neighbors | `Dataset`  |        ✅         |        x        |         x         |          x           |        x         |               x                |
| [`Kraskov`](@ref)            | Nearest neighbors | `Dataset`  |        ✅         |        x        |         x         |          x           |        x         |               x                |
| [`Zhu`](@ref)                | Nearest neighbors | `Dataset`  |        ✅         |        x        |         x         |          x           |        x         |               x                |
| [`ZhuSingh`](@ref)           | Nearest neighbors | `Dataset`  |        ✅         |        x        |         x         |          x           |        x         |               x                |
| [`Vasicek`](@ref)            | Order statistics  | `Vector`   |        ✅         |        x        |         x         |          x           |        x         |               x                |
| [`Ebrahimi`](@ref)           | Order statistics  | `Vector`   |        ✅         |        x        |         x         |          x           |        x         |               x                |
| [`Correa`](@ref)             | Order statistics  | `Vector`   |        ✅         |        x        |         x         |          x           |        x         |               x                |
| [`AlizadehArghami`](@ref)    | Order statistics  | `Vector`   |        ✅         |        x        |         x         |          x           |        x         |               x                |

```@docs
DiffEntropyEst
```

```@docs
Kraskov
KozachenkoLeonenko
Zhu
ZhuSingh
Vasicek
AlizadehArghami
Ebrahimi
Correa
```

## Convenience functions

In this subsection we expand documentation strings of "entropy names" that are used commonly in the literature, such as "permutation entropy". As we made clear in [API & terminology](@ref), these are just the existing Shannon entropy with a particularly chosen probability estimator. We have only defined convenience functions for the most used names, and arbitrary more specialized convenience functions can be easily defined in a couple lines of code.

```@docs
entropy_permutation
entropy_spatial_permutation
entropy_wavelet
entropy_dispersion
```
