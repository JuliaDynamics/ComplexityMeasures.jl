# [Probabilities

## Probabilities API

The probabilities API is defined by

- [`ProbabilitiesEstimator`](@ref)
- [`probabilities`](@ref)
- [`probabilities_and_outcomes`](@ref)

and related functions that you will find in the following documentation block:

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

## [Overview of probabilities estimators](@id probabilities_estimators)

Any of the following estimators can be used with [`probabilities`](@ref)
(in the column "input data"  it is assumed that the `eltype` of the input is `<: Real`).

| Estimator                                   | Principle                   | Input data          |
|:--------------------------------------------|:----------------------------|:--------------------|
| [`CountOccurrences`](@ref)                  | Count of unique elements    | `Any` |
| [`ValueHistogram`](@ref)                    | Binning (histogram)         | `Vector`, `Dataset` |
| [`TransferOperator`](@ref)                  | Binning (transfer operator) | `Vector`, `Dataset` |
| [`NaiveKernel`](@ref)                       | Kernel density estimation   | `Dataset`           |
| [`SymbolicPermutation`](@ref)               | Ordinal patterns            | `Vector`, `Dataset` |
| [`SymbolicWeightedPermutation`](@ref)       | Ordinal patterns            | `Vector`, `Dataset` |
| [`SymbolicAmplitudeAwarePermutation`](@ref) | Ordinal patterns            | `Vector`, `Dataset` |
| [`SpatialSymbolicPermutation`](@ref)        | Ordinal patterns in space   | `Array` |
| [`Dispersion`](@ref)                        | Dispersion patterns         | `Vector`            |
| [`SpatialDispersion`](@ref)                 | Dispersion patterns in space        | `Array` |
| [`Diversity`](@ref)                         | Cosine similarity           | `Vector`            |
| [`WaveletOverlap`](@ref)                    | Wavelet transform           | `Vector`            |
| [`PowerSpectrum`](@ref)                     | Fourier spectra             | `Vector` |

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
SpatialDispersion
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
