# [Entropies](@id entropies)

```@docs
entropy
Entropy
```

## Generalized entropies

```@docs
Shannon
Renyi
Tsallis
Curado
StretchedExponential
```

## Estimation

### Discrete entropies

Discrete entropies are just simple functions (sums, actually) of
probability mass functions [(pmf)](https://en.wikipedia.org/wiki/Probability_mass_function),
which you can estimate using [`ProbabilitiesEstimator`](@ref)s.

Any [`ProbabilitiesEstimator`](@ref) may therefore be used as a naive plug-in estimator
for discrete [`entropy`](@ref). No bias correction is currently applied to any of the
discrete estimators.

Tables scroll sideways, so are best viewed on a large screen.

| Estimator                                   | Principle                   | Input data          | [`Shannon`](@ref) | [`Renyi`](@ref) | [`Tsallis`](@ref) | [`Curado`](@ref) | [`StretchedExponential`](@ref) |
| ------------------------------------------- | --------------------------- | ------------------- | :---------------: | :-------------: | :---------------: | :--------------: | :----------------------------: |
| [`CountOccurrences`](@ref)                  | Frequencies                 | `Vector`, `Dataset` |        ✅         |       ✅        |        ✅         |        ✅        |               ✅               |
| [`ValueHistogram`](@ref)                    | Binning (histogram)         | `Vector`, `Dataset` |        ✅         |       ✅        |        ✅         |        ✅        |               ✅               |
| [`TransferOperator`](@ref)                  | Binning (transfer operator) | `Vector`, `Dataset` |        ✅         |       ✅        |        ✅         |        ✅        |               ✅               |
| [`NaiveKernel`](@ref)                       | Kernel density estimation   | `Dataset`           |        ✅         |       ✅        |        ✅         |        ✅        |               ✅               |
| [`LocalLikelihood`](@ref)                   | Local likelihood Estimation | `Dataset`           |        ✅         |       ✅        |        ✅         |        ✅        |               ✅               |
| [`SymbolicPermutation`](@ref)               | Ordinal patterns            | `Vector`            |        ✅         |       ✅        |        ✅         |        ✅        |               ✅               |
| [`SymbolicWeightedPermutation`](@ref)       | Ordinal patterns            | `Vector`            |        ✅         |       ✅        |        ✅         |        ✅        |               ✅               |
| [`SymbolicAmplitudeAwarePermutation`](@ref) | Ordinal patterns            | `Vector`            |        ✅         |       ✅        |        ✅         |        ✅        |               ✅               |
| [`Dispersion`](@ref)                        | Dispersion patterns         | `Vector`            |        ✅         |       ✅        |        ✅         |        ✅        |               ✅               |
| [`Diversity`](@ref)                         | Cosine similarity           | `Vector`            |        ✅         |       ✅        |        ✅         |        ✅        |               ✅               |
| [`WaveletOverlap`](@ref)                    | Wavelet transform           | `Vector`            |        ✅         |       ✅        |        ✅         |        ✅        |               ✅               |
| [`PowerSpectrum`](@ref)                     | Fourier spectra             | `Vector`, `Dataset` |        ✅         |       ✅        |        ✅         |        ✅        |               ✅               |

### Continuous/differential entropies

The following estimators are *differential* entropy estimators, and can also be used
with [`entropy`](@ref). Differential) entropies are functions of *integrals*, and usually
rely on estimating some density functional.

Each [`EntropyEstimator`](@ref)s uses a specialized technique to approximating relevant
densities/integrals, and is often tailored to one or a few types of generalized entropy.
For example, [`Kraskov`](@ref) estimates the [`Shannon`](@ref) entropy, while
[`LeonenkoProzantoSavani`](@ref) estimates [`Shannon`](@ref), [`Renyi`](@ref), and
[`Tsallis`](@ref) entropies.

| Estimator                    | Principle         | Input data | [`Shannon`](@ref) | [`Renyi`](@ref) | [`Tsallis`](@ref) | [`Curado`](@ref) | [`StretchedExponential`](@ref) |
| ---------------------------- | ----------------- | ---------- | :---------------: | :-------------: | :---------------: | :--------------: | :----------------------------: |
| [`KozachenkoLeonenko`](@ref) | Nearest neighbors | `Dataset`  |        ✅         |        x        |         x         |        x         |               x                |
| [`Kraskov`](@ref)            | Nearest neighbors | `Dataset`  |        ✅         |        x        |         x         |        x         |               x                |
| [`Zhu`](@ref)                | Nearest neighbors | `Dataset`  |        ✅         |        x        |         x         |        x         |               x                |
| [`ZhuSingh`](@ref)           | Nearest neighbors | `Dataset`  |        ✅         |        x        |         x         |        x         |               x                |
| [`Vasicek`](@ref)            | Order statistics  | `Vector`   |        ✅         |        x        |         x         |        x         |               x                |
| [`Ebrahimi`](@ref)           | Order statistics  | `Vector`   |        ✅         |        x        |         x         |        x         |               x                |
| [`Correa`](@ref)             | Order statistics  | `Vector`   |        ✅         |        x        |         x         |        x         |               x                |
| [`AlizadehArghami`](@ref)    | Order statistics  | `Vector`   |        ✅         |        x        |         x         |        x         |               x                |

```@docs
EntropyEstimator
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
