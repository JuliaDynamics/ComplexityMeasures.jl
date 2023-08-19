# Probabilities

!!! note
    Please be sure you have read the [Terminology](@ref terminology) section before going through the API here.


## [Outcome spaces (discretization)](@id outcome_spaces)

```@docs
OutcomeSpace
outcomes
outcome_space
total_outcomes
missing_outcomes
```

### Count occurrences

```@docs
CountOccurrences
```

### Histograms

```@docs
ValueHistogram
AbstractBinning
RectangularBinning
FixedRectangularBinning
```

### Symbolic permutations

```@docs
OrdinalPatterns
WeightedOrdinalPatterns
AmplitudeAwareOrdinalPatterns
```

### Dispersion patterns

```@docs
Dispersion
```

### Transfer operator

```@docs
TransferOperator
```

#### Utility methods/types

```@docs
InvariantMeasure
invariantmeasure
transfermatrix
```

### Kernel density

```@docs
NaiveKernel
```

### Timescales

```@docs
WaveletOverlap
PowerSpectrum
```

### Diversity

```@docs
Diversity
```

### Spatial outcome spaces

```@docs
SpatialOrdinalPatterns
SpatialDispersion
```

## Probabilities API

The probabilities API is defined by

- [`OutcomeSpace`](@ref), which defines a set of possible outcomes onto which input data
    are mapped (specifies a discretization).
- [`ProbabilitiesEstimator`](@ref), which maps observed (pseudo-)counts of outcomes to
    probabilities.
- [`probabilities`](@ref) and [`allprobabilities`](@ref)
- [`probabilities_and_outcomes`](@ref) and [`allprobabilities_and_outcomes`](@ref)

and related functions that you will find in the following documentation blocks:

```@docs
Probabilities
probabilities
probabilities_and_outcomes
probabilities!
allprobabilities
allprobabilities_and_outcomes
counts_and_outcomes
allcounts_and_outcomes
counts
is_counting_based
```

## [Probability estimators](@id probability_estimators)

```@docs
ProbabilitiesEstimator
RelativeAmount
Bayes
Shrinkage
```

## [Encodings API](@id encodings)

Some [`OutcomeSpace`](@ref)s first "encode" input data into an intermediate representation
indexed by the positive integers. This intermediate representation is called an "encoding".

The encodings API is defined by:

- [`Encoding`](@ref)
- [`encode`](@ref)
- [`decode`](@ref)

```@docs
Encoding
encode
decode
```

### Available encodings

```@docs
OrdinalPatternEncoding
GaussianCDFEncoding
RectangularBinEncoding
```
