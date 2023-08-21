# [Probabilities](@ref)

!!! note
    Be sure you have gone through the [Tutorial](@ref) before going through the API here to have a good idea of the terminology used in ComplexityMeasures.jl.

ComplexityMeasures.jl implements an interface for probabilities that exactly follows the mathematically rigorous formulation of [probability spaces](https://en.wikipedia.org/wiki/Probability_space).
Probability spaces are formalized by an [`OutcomeSpace`](@ref) $\Omega$, and the functions [`counts`](@ref) and [`probabilities`](@ref) as well as derivative functions such as [`allprobabilities_and_outcomes`](@ref).
The mathematical formulation of probabilities spaces is further enhanced by [`ProbabilitiesEstimator`](@ref) and its subtypes, which may correct theoretically known biases when estimating probabilities from finite data.

In reality, probabilities can be either discrete ([mass functions](https://en.wikipedia.org/wiki/Probability_mass_function)) or continuous ([density functions](https://en.wikipedia.org/wiki/Probability_density_function)).
Currently in ComplexityMeasures.jl, only probability mass functions (i.e., countable $\Omega$) are implemented explicitly. Quantities that are estimated from probability density functions (i.e., uncountable $\Omega$) also exist and are implemented in ComplexityMeasures.jl. However, these are estimated by a one-step processes without the intermediate estimation of probabilities.

If $\Omega$ is countable, the process of estimating the outcomes from input data is also called _discretization_ of the input data.

## [Outcome spaces](@id outcome_spaces)

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

## Probabilities

```@docs
Probabilities
probabilities
probabilities_and_outcomes
probabilities!
allprobabilities
allprobabilities_and_outcomes
```

## Counts

```@docs
counts
counts_and_outcomes
allcounts_and_outcomes
is_counting_based
```

## [Probability estimators](@id probability_estimators)

```@docs
ProbabilitiesEstimator
RelativeAmount
BayesianRegularization
Shrinkage
```

## [Encodings API](@id encodings)

Count-based [`OutcomeSpace`](@ref)s first "encode" input data into an intermediate representation indexed by the positive integers.
This intermediate representation is called an "encoding".

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
