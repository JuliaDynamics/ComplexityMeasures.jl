# ComplexityMeasures.jl

```@docs
ComplexityMeasures
```

## Content and terminology

!!! note
    The documentation here follows (loosely) chapter 5 of
    [Nonlinear Dynamics](https://link.springer.com/book/10.1007/978-3-030-91032-7),
    Datseris & Parlitz, Springer 2022.

Before exploring the features of ComplexityMeasures.jl, it is useful to read through this terminology section to have an idea of the content, but also what the content means how it is called within the context of this package!

In the literature, the term "entropy" is used (and abused) in multiple contexts.
The API and documentation of ComplexityMeasures.jl aim to clarify some aspects of its usage, and to provide a simple way to obtain probabilities, entropies, or other complexity measures.

### Probabilities

ComplexityMeasures and other complexity measures are typically computed based on _probability distributions_.
These can be obtained from input data in a plethora of different ways.
The central API function that returns a probability distribution (or more precisely a probability mass function) is [`probabilities`](@ref), which takes in a subtype of [`ProbabilitiesEstimator`](@ref) to specify how the probabilities are computed.
All available estimators can be found in the [estimators page](@ref probabilities_estimators).

### Entropies

Entropy is an established concept in statistics, information theory, and nonlinear dynamics.
However it is also an umbrella term that may mean several computationally, and sometimes even fundamentally, different quantities.
In ComplexityMeasures.jl, we provide the generic function [`entropy`](@ref) that tries to both clarify the disparate "entropy concepts", while unifying them under a common interface that highlights the modular nature of the word "entropy".

In the typical case, computing an entropy means computing a _discrete_ entropy, which boils down to two simple steps: first estimating a probability distribution, and then applying one of the so-called "generalized entropy" definitions to the distributions.

A crucial thing to clarify is that in the nonlinear dynamics literature many quantities that are named as entropies (e.g., permutation entropy [`entropy_permutation`](@ref), wavelet entropy [`entropy_wavelet`](@ref), etc.), are _not really new entropies_. They are the good old Shannon entropy ([`Shannon`](@ref)) but calculated with _new probabilities estimators_.
So even if the naming ("wavelet entropy") sounds like a name for a new entropy, these approaches actually devise a new way to calculate probabilities from data, and then plug those probabilities into formal entropy formulas such as the Shannon entropy. The probabilities estimators are of course smartly created so that they elegantly highlight important aspects of the data relevant to complexity.

These names such as "permutation entropy" are commonplace, and so in ComplexityMeasures.jl we provide convenience functions like [`entropy_permutation`](@ref). However, it should be noted that these functions really aren't anything more than 2-lines-of-code wrappers that call [`entropy`](@ref) with the appropriate [`ProbabilitiesEstimator`](@ref).

What are _genuinely different entropies_ are different definitions of entropy. And there are a lot of them! E.g., [`Renyi`](@ref) or [`Tsallis`](@ref). These different definitions can be found in [`EntropyDefinition`](@ref).

In addition to the above, we also provide [`DifferentialEntropyEstimator`](@ref)s.
These compute entropies via alternate means, without explicitly computing some
probability distribution. For example, the [`Correa`](@ref) estimator computes the Shannon differential entropy using order statistics.
Differential entropies are functions of _integrals_, and usually
rely on estimating some density functional.


### Other complexity measures

Other complexity measures, which strictly speaking don't compute entropies, and may or may not explicitly compute probability distributions, are found in
[Complexity measures](@ref) page.
This includes measures like sample entropy and approximate entropy.

## [Input data for ComplexityMeasures.jl](@id input_data)

The input data type typically depend on the probability estimator chosen.
In general though, the standard DynamicalSystems.jl approach is taken and as such we have three types of input data:

- _Timeseries_, which are `AbstractVector{<:Real}`, used in e.g. with [`WaveletOverlap`](@ref).
- _Multi-variate timeseries, or datasets, or state space sets_, which are [`Dataset`](@ref), used e.g. with [`NaiveKernel`](@ref).
- _Spatial data_, which are higher dimensional standard `Array`s, used e.g. with  [`SpatialSymbolicPermutation`](@ref).

```@docs
Dataset
```