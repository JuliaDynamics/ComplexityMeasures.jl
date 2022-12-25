# Entropies.jl

```@docs
Entropies
```

!!! info
    You are reading the development version of the documentation of Entropies.jl,
    that will become version 2.0.

## Terminology

!!! note
    The documentation here follows (loosely) chapter 5 of
    [Nonlinear Dynamics](https://link.springer.com/book/10.1007/978-3-030-91032-7),
    Datseris & Parlitz, Springer 2022.

In the literature, the term "entropy" is used (and abused) in multiple contexts.
The API and documentation of Entropies.jl aim to clarify some aspects of its usage, and to provide a simple way to obtain probabilities, entropies, or other complexity measures.

### Probabilities

Entropies and other complexity measures are typically computed based on _probability distributions_.
These can be obtained from input data in a plethora of different ways.
The central API function that returns a probability distribution (or more precisely a probability mass function) is [`probabilities`](@ref), which takes in a subtype of [`ProbabilitiesEstimator`](@ref) to specify how the probabilities are computed.
All available estimators can be found in the [estimators page](@ref probabilities_estimators).

### Entropies

EntropyDefinition is an established concept in statistics, information theory, and nonlinear dynamics.
However it is also an umbrella term that may mean several computationally different quantities.
In Entropies.jl, we provide the generic function [`entropy`](@ref) that tries to both clarify the disparate "entropy concepts", while unifying them under a common interface that highlights the modular nature of the word "entropy".

Most of the time, computing an entropy boils down to two simple steps: first estimating a probability distribution, and then applying one of the so-called "generalized entropy" formulas to the distributions.
Thus, any of the implemented [probabilities estimators](@ref probabilities_estimators) can be used to compute generalized entropies.

!!! tip "There aren't many entropies, really."
    A crucial thing to clarify is that many quantities that are named as entropies (e.g., permutation entropy [`entropy_permutation`](@ref), wavelet entropy [`entropy_wavelet`](@ref), etc.), are _not really new entropies_. They are new probabilities estimators. They simply devise a new way to calculate probabilities from data, and then plug those probabilities into formal entropy formulas such as the Shannon entropy. The probabilities estimators are smartly created so that they elegantly highlight important aspects of the data relevant to complexity.

    These names are commonplace, and so in Entropies.jl we provide convenience functions like [`entropy_wavelet`](@ref). However, it should be noted that these functions really aren't anything more than 2-lines-of-code wrappers that call [`entropy`](@ref) with the appropriate [`ProbabilitiesEstimator`](@ref).

    In addition to `ProbabilitiesEstimators`, we also provide [`DiffEntropyEst`](@ref)s,
    which compute entropies via alternate means, without explicitly computing some
    probability distribution. Differential/continuous entropy, for example, is computed
    using a dedicated [`DiffEntropyEst`](@ref). For example, the [`Kraskov`](@ref)
    estimator computes Shannon differential entropy via a nearest neighbor algorithm, while
    the [`Correa`](@ref) estimator computes Shannon differential entropy using order statistics.

### Other complexity measures

Other complexity measures, which strictly speaking don't compute entropies, and may or may not explicitly compute probability distributions, are found in
[Complexity measures](@ref) page.
This includes measures like sample entropy and approximate entropy.

## [Input data for Entropies.jl](@id input_data)

The input data type typically depend on the probability estimator chosen.
In general though, the standard DynamicalSystems.jl approach is taken and as such we have three types of input data:

- _Timeseries_, which are `AbstractVector{<:Real}`, used in e.g. with [`WaveletOverlap`](@ref).
- _Multi-dimensional timeseries, or datasets, or state space sets_, which are [`Dataset`](@ref), used e.g. with [`NaiveKernel`](@ref).
- _Spatial data_, which are higher dimensional standard `Array`s, used e.g. with  [`SpatialSymbolicPermutation`](@ref).

```@docs
Dataset
```