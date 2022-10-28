# [Probabilities](@id probabilities_estimators)

## Probabilities API

```@docs
ProbabilitiesEstimator
```

!!! note "A more formal treatment"
    You may notice that we don't use the word "event" at all in this package. Here's why.

    A [**probability space**](https://en.wikipedia.org/wiki/Probability_space) is a
    triple `P(Ω, A, ℙ)` that models a random process/experiment. Here, `Ω` is the *set* of
    possible outcomes of the experiment
    ([sample space](https://en.wikipedia.org/wiki/Sample_space)), as described above.
    The set `A` is a [σ-algebra](https://en.wikipedia.org/wiki/%CE%A3-algebra) on `Ω`
    containing events we consider relevant. Finally, `ℙ : A → [0, 1]` is a
    set function that assigns a real number `p` to every event, such that `∑ᵢ(pᵢ) = 1`,
    i.e. `ℙ` assigns probabilities to the events we consider relevant.

    What are "relevant events"? Well, an "event" can be any subset of `Ω`, 
    including compound events (i.e. consisting of multiple of the elementary outcomes in 
    `Ω`), like `{ω₁, ω₄, ω₉}` or `{ω₂, ω₃}`. In fact, the event space for a probability 
    space is often taken to be the power set of `Ω`. However, in Entropies.jl, currently 
    implemented functions of probabilities consider the singleton elements in `A` only.
    These events are precisely the elementary outcomes `ωᵢ ∈ Ω`. Therefore, we consistently 
    use "outcome" instead of "event".

    If desired, the probabilities of compound events can easily be obtained from the 
    computed [`Probabilities`](@ref) under relevant assumptions about 
    dependence/independence of the outcomes.

```@docs
Probabilities
probabilities
probabilities!
probabilities_and_outcomes
total_outcomes
```

## Count occurrences (counting)

```@docs
CountOccurrences
```

## Visitation frequency (histograms)

```@docs
ValueHistogram
RectangularBinning
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
