# Probabilities

## Probabilities API

The probabilities API is defined by

- [`ProbabilitiesEstimator`](@ref)
- [`probabilities`](@ref)
- [`probabilities_and_outcomes`](@ref)

and related functions that you will find in the following documentation blocks:

### Probabilitities

```@docs
ProbabilitiesEstimator
probabilities
probabilities!
Probabilities
```

### Outcomes

```@docs
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
| [`ValueHistogram`](@ref)                    | Binning (histogram)         | `Vector`, `StateSpaceSet` |
| [`TransferOperator`](@ref)                  | Binning (transfer operator) | `Vector`, `StateSpaceSet` |
| [`NaiveKernel`](@ref)                       | Kernel density estimation   | `StateSpaceSet`           |
| [`SymbolicPermutation`](@ref)               | Ordinal patterns            | `Vector`, `StateSpaceSet` |
| [`SymbolicWeightedPermutation`](@ref)       | Ordinal patterns            | `Vector`, `StateSpaceSet` |
| [`SymbolicAmplitudeAwarePermutation`](@ref) | Ordinal patterns            | `Vector`, `StateSpaceSet` |
| [`SpatialSymbolicPermutation`](@ref)        | Ordinal patterns in space   | `Array` |
| [`Dispersion`](@ref)                        | Dispersion patterns         | `Vector`            |
| [`SpatialDispersion`](@ref)                 | Dispersion patterns in space  | `Array` |
| [`Diversity`](@ref)                         | Cosine similarity           | `Vector`            |
| [`WaveletOverlap`](@ref)                    | Wavelet transform           | `Vector`            |
| [`PowerSpectrum`](@ref)                     | Fourier transform           | `Vector` |

## Count occurrences

```@docs
CountOccurrences
```

## Histograms

```@docs
ValueHistogram
RectangularBinning
FixedRectangularBinning
```

## Symbolic permutations

```@docs
SymbolicPermutation
SymbolicWeightedPermutation
SymbolicAmplitudeAwarePermutation
```

## Dispersion patterns

```@docs
Dispersion
```

## Transfer operator

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

## Spatial estimators

```@docs
SpatialSymbolicPermutation
SpatialDispersion
```
