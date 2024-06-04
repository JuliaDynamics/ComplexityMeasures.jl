# ComplexityMeasures.jl

```@docs
ComplexityMeasures
```

## Latest news

ComplexityMeasures.jl has been updated to v3!

The software has been massively improved and its core principles were
redesigned to be extendable, accessible, and more closely based
on the rigorous mathematics of probabilities and entropies.

For more details of this new release, please see our [announcement
post on discourse](https://discourse.julialang.org/t/complexitymeasures-jl-v3-a-mathematically-rigorous-software-for-probability-entropy-and-complexity/108562)
or the central [Tutorial](@ref) of the v3 documentation.

In this v3 many concepts were renamed, but there is no formally
breaking change. Everything that changed has been deprecated
and is backwards compatible. You can see the
[CHANGELOG.md](https://github.com/JuliaDynamics/ComplexityMeasures.jl/blob/v3.0.0/CHANGELOG.md) for more details!

## Documentation contents

* Before anything else, we recommend users to go through our overarching [Tutorial](@ref), which teaches not only central API functions, but also terminology and crucial core concepts:
* [Probabilities](@ref) lists all outcome spaces and probabilities estimators.
* [Information measures](@ref information_measures) lists all implemented information measure definitions and estimators (both discrete and differential).
* [Complexity measures](@ref complexity_measures) lists all implemented complexity measures that are not functionals of probabilities (unlike information measures).
* The [Examples](@ref examples) page lists dozens of runnable example code snippets along with their outputs.

## [Input data for ComplexityMeasures.jl](@id input_data)

The input data type typically depend on the outcome space chosen.
In general though, the standard DynamicalSystems.jl approach is taken and as such we have three types of input data:

- *Timeseries*, which are `AbstractVector{<:Real}`, used in e.g. with [`WaveletOverlap`](@ref).
- *Multi-variate timeseries, or datasets, or state space sets*, which are [`StateSpaceSet`](@ref)s, used e.g. with [`NaiveKernel`](@ref). The short syntax `SSSet` may be used instead of `StateSpaceSet`.
- *Spatial data*, which are higher dimensional standard `Array`s, used e.g. with  [`SpatialOrdinalPatterns`](@ref).

```@docs
StateSpaceSet
```

## Total entropy/information/complexity measures

ComplexityMeasures.jl offers thousands of measures computable right out of the box.
To see an exact number of how many, see this [calculation page](@ref total_measures).
