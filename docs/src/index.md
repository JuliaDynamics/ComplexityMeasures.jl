# Entropies.jl
```@docs
Entropies
```

## API & terminology

!!! note
    The documentation here follows (loosely) chapter 5 of
    [Nonlinear Dynamics](https://link.springer.com/book/10.1007/978-3-030-91032-7),
    Datseris & Parlitz, Springer 2022.


In the literature, the term "entropy" is used (and abused) in multiple contexts.
The API and documentation of Entropies.jl aim to clarify some aspects and
provide a simple way to obtain probabilities, entropies, or other complexity measures.

### Probabilities
Entropies and other complexity measures are typically computed based on _probability distributions_.
These are obtained from [Input data](@ref) by a plethora of different ways.
The central API function that returns a probability distribution (actual, just a vector of probabilities) is [`probabilities`](@ref), which takes in a subtype of [`ProbabilityEstimator`](@ref) to specify how the probabilities are computed.
All estimators available in Entropies.jl can be found in the [estimators page](@ref estimators).

### Entropies
Entropy is an established concept in statistics, information theory, and nonlinear dynamics. However it is also an umbrella term that may mean several computationally different quantities.

[Generalized entropies](@ref) are theoretically well-founded and in Entropies.jl we have the
- Rényi entropy [`entropy_renyi`](@ref).
- Tsallis entropy [`entropy_tsallis`](@ref).
- Shannon entropy [`entropy_shannon`](@ref), which is just a subcase of either of the above two.

Computing such an entropy most of the time boils down to two simple steps: first estimating a probability distribution, and then applying one of the generalized entropy formulas to the distributions.
Thus, any of the implemented [probabilities estimators](@ref estimators) can be used to compute generalized entropies.


!!! tip "There aren't many entropies, really."
    A crucial thing to clarify, is that many quantities that are named as entropies (e.g., permutation entropy ([`entropy_permutation`](@ref)), the wavelet entropy [`entropy_wavelet`](@ref), etc.), are _not really new entropies_. They are in fact new probability estimators. They simply devise a new way to calculate probabilities from data, and then plug those probabilities into formal entropy formulas such as the Shannon entropy. While in Entropies.jl we provide convenience functions like [`entropy_wavelet`](@ref), they really aren't anything more than 3-lines-of-code wrappers that call [`entropy_shannon`](@ref) with the appropriate [`ProbabilityEstimator`](@ref).

    There are only a few exceptions to this rule, which are quantities that are able to compute Shannon entropies via alternate means, without explicitly computing some probability distributions, such as [TODO ADD EXAMPLE].


### Complexity measures
Other complexity measures, which strictly speaking don't compute entropies, and may or may not explicitly compute probability distributions, appear in the [Complexity measures](@ref) section.


## Input data
The input data type typically depend on the probability estimator chosen. In general though, the standard DynamicalSystems.jl approach is taken and as such we have three types of input data:

- _Timeseries_, which are `AbstractVector{<:Real}`, used in e.g. with [`WaveletOverlap`](@ref).
- _Multi-dimensional timeseries, or datasets, or state space sets_, which are `Dataset`, used e.g. with [`NaiveKernel`](@ref).
- _Spatial data_, which are higher dimensional standard `Array`, used e.g. with  [`SpatialSymbolicPermutation`](@ref).
