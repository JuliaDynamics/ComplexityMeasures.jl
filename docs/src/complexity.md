# [Complexity measures](@id complexity_measures)

!!! note
    Be sure you have gone through the [Tutorial](@ref) before going through the API here to have a good idea of the terminology used in ComplexityMeasures.jl.

## Complexity measures API

The complexity measure API is defined by the [`complexity`](@ref) function, which may take as an input an [`ComplexityEstimator`](@ref). The function [`complexity_normalized`](@ref) is also useful.

```@docs
complexity
complexity_normalized
ComplexityEstimator
```

## Approximate entropy

```@docs
ApproximateEntropy
```

## Sample entropy

```@docs
SampleEntropy
```

## Missing dispersion patterns

```@docs
MissingDispersionPatterns
```

## Reverse dispersion entropy

```@docs
ReverseDispersion
```

## Statistical complexity

```@docs
StatisticalComplexity
entropy_complexity
entropy_complexity_curves
```

## Lempel-Ziv complexity

```@docs
LempelZiv76
```
