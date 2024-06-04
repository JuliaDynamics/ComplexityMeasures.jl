# Multiscale

## Introduction

Multiscale complexity analysis is pervasive in the nonlinear time series analysis
literature. Although their names, like "refined composite multiscale dispersion entropy",  
might seem daunting, they're actually conceptually very simple. A multiscale complexity
measure is just any regular complexity measure
computed on several gradually more coarse-grained samplings of the input data
([example](@ref multiscale_example)).

We've generalized this type of analysis to work with any information measure, discrete or
differential, and to any other type of complexity measure we implement.

## Multiscale API

The multiscale API is defined by the functions

- [`multiscale`](@ref)
- [`multiscale_normalized`](@ref)
- [`downsample`](@ref)

which dispatch any of the [`MultiScaleAlgorithm`](@ref)s listed below.

```@docs
MultiScaleAlgorithm
RegularDownsampling
CompositeDownsampling
downsample
multiscale
multiscale_normalized
```

## Available literature methods

A non-exhaustive list of literature methods, and the syntax to compute them, are listed
below. Please open an issue or make a pull-request to
[ComplexityMeasures.jl](https://github.com/JuliaDynamics/ComplexityMeasures.jl) if you
find a literature method missing from this list, or if you publish a paper based on some
new multiscale combination.

| Method                                                | Syntax example                                                                 | Reference          |
| ----------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------ |
| Refined composite multiscale dispersion entropy       | `multiscale(CompositeDownsampling(), Dispersion(), est, x, normalized = true)` | [Azami2017](@cite) |
| Multiscale sample entropy (first moment)              | `multiscale(RegularDownsampling(f = mean), SampleEntropy(x), x)`               | [Costa2002](@cite) |
| Generalized multiscale sample entropy (second moment) | `multiscale(RegularDownsampling(f = std), SampleEntropy(x),  x)`               | [Costa2015](@cite) |
