# [Probabilities](@id probabilities_estimators)

## Probabilities API

The probabilities API is defined by

- [`ProbabilitiesEstimator`](@ref)
- [`probabilities`](@ref)
- [`probabilities_and_outcomes`](@ref)

```@docs
ProbabilitiesEstimator
probabilities
probabilities!
Probabilities
probabilities_and_outcomes
outcomes
outcome_space
total_outcomes
missing_outcomes
```

## Overview

Any of the following estimators can be used with [`probabilities`](@ref).

| Estimator                                   | Principle                   | Input data          |
| ------------------------------------------- | --------------------------- | ------------------- |
| [`CountOccurrences`](@ref)                  | Frequencies                 | `Vector`, `Dataset` |
| [`ValueHistogram`](@ref)                    | Binning (histogram)         | `Vector`, `Dataset` |
| [`TransferOperator`](@ref)                  | Binning (transfer operator) | `Vector`, `Dataset` |
| [`NaiveKernel`](@ref)                       | Kernel density estimation   | `Dataset`           |
| [`SymbolicPermutation`](@ref)               | Ordinal patterns            | `Vector`            |
| [`SymbolicWeightedPermutation`](@ref)       | Ordinal patterns            | `Vector`            |
| [`SymbolicAmplitudeAwarePermutation`](@ref) | Ordinal patterns            | `Vector`            |
| [`Dispersion`](@ref)                        | Dispersion patterns         | `Vector`            |
| [`Diversity`](@ref)                         | Cosine similarity           | `Vector`            |
| [`WaveletOverlap`](@ref)                    | Wavelet transform           | `Vector`            |
| [`PowerSpectrum`](@ref)                     | Fourier spectra             | `Vector`, `Dataset` |

## Count occurrences (counting)

```@docs
CountOccurrences
```

## Visitation frequency (histograms)

```@docs
ValueHistogram
RectangularBinning
FixedRectangularBinning
```

## Permutation (symbolic)

```@docs
SymbolicPermutation
SymbolicWeightedPermutation
SymbolicAmplitudeAwarePermutation
SpatialSymbolicPermutation
```

## Dispersion (symbolic)

```@docs
Dispersion
```

## Transfer operator (binning)

```@docs
TransferOperator
```

### Utility methods/types

```@docs
InvariantMeasure
invariantmeasure
transfermatrix
```

## Kernel density

```@docs
NaiveKernel
```

## Timescales

```@docs
WaveletOverlap
PowerSpectrum
```

## Diversity

```@docs
Diversity
```
