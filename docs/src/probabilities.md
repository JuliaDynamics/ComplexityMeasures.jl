# [Probabilities](@id estimators)

For categorical or integer-valued data, probabilities can be estimated by directly counting relative frequencies of data elements. For such data, use `probabilities(x::Array_or_Dataset) → p::Probabilities`.

More advanced estimators computing probabilities by first either discretizing, symbolizing or transforming the data in a way that quantifies some useful properties about the underlying data (e.g. visitation frequencies, wavelet energies, or permutation patterns), from which probability distributions can be estimated. Use `probabilities(x::Array_or_Dataset, est::ProbabilitiesEstimator)` in combination with any of the estimators listed below.

```@docs
probabilities
probabilities!
Probabilities
ProbabilitiesEstimator
```

## Count occurrences (counting)

```@docs
CountOccurrences
```

## Permutation (symbolic)

```@docs
SymbolicPermutation
SpatialSymbolicPermutation
```

## Dispersion (symbolic)

```@docs
Dispersion
ReverseDispersion
```

## Visitation frequency (binning)

```@docs
VisitationFrequency
```

### Specifying binning/boxes

```@docs
RectangularBinning
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
