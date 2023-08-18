# ComplexityMeasures.jl

```@docs
ComplexityMeasures
```

## [Content and terminology](@id terminology)

!!! note
    The documentation here was initially inspired by chapter 5 of
    [Nonlinear Dynamics](https://link.springer.com/book/10.1007/978-3-030-91032-7),
    Datseris & Parlitz, Springer 2022.

Before exploring the features of ComplexityMeasures.jl, it is useful to read through this terminology section. In the scientific literature, words like **probabilities**, **entropy**, and **complexity measure** are used (and abused) in multiple contexts, and are often used interchangeably to describe similar concepts. The API and documentation of ComplexityMeasures.jl aim to clarify the meaning and usage of these words, and to provide simple ways to obtain probabilities, information measures, or other complexity measures from input data.

For ComplexityMeasures.jl, we use the generic term "complexity measure" for any complexity measure that is not a direct functional of probabilities.
"Information measure" is used exclusively about measures that are functionals of probability mass functions or probability density functions, for example entropies.
We believe the general nonlinear dynamics community agrees with our take, as most papers that introduce different entropy flavors call them complexity measures. Example: *"Permutation Entropy: A Natural Complexity Measure for Time Series"* (Brandt and Pompe, 2002).

### Probabilities and Outcome Spaces

Information measures, and some other complexity measures, are are typically computed based on **probability distributions** derived from input data.
Probabilities can be either discrete ([mass functions](https://en.wikipedia.org/wiki/Probability_mass_function)) or continuous ([density functions](https://en.wikipedia.org/wiki/Probability_density_function)).
ComplexityMeasures.jl implements an interface for probabilities that exactly follows the mathematically rigorous formulation of [probability spaces](https://en.wikipedia.org/wiki/Probability_space).
In order to derive probabilities from data, an **outcome space** needs to be defined: a way to transform data into elements $\omega$ of an outcome space $\omega \in \Omega$, and assign probabilities to each outcome $p(\omega)$, such that $p(\Omega)=1$. $\omega$ are called _outcomes_ or _events_.

If $\Omega$ is countable, this process is also called _discretization_ of the input data. During discretization, each element of input data $\chi \in x$ is [`encode`](@ref)d (discretized) into an element of the outcome space $\omega \in \Omega$.
Currently in ComplexityMeasures.jl, only countable $\Omega$ are implemented explicitly. Quantities that are estimated from probability density functions (i.e., uncountable $\Omega$) also exist. However, these are estimated by a one-step processes without the intermediate estimation of probabilities.

Once an outcome space (subtype of [`OutcomeSpace`](@ref)) is defined in ComplexityMeasures.jl, probabilities are estimated from input data using the [`probabilities`](@ref) function.
The simplest way to do this is simply to assign a probability to each outcome as its _relative count_ with respect to the total outcomes (i.e., normalizing the output of the [`counts`](@ref) function). This is also the default way.
However, more advanced versions to estimate probabilities exist, that may account for known estimation biases. These **probabilities estimators** are subtypes of [`ProbabilitiesEstimator`](@ref).

### Information measures

Within ComplexityMeasures.jl, we have taken the pragmatic choice to label all measures that are _explicit functionals of probability mass or probability density functions_ as **information measures**, even though they might not be labelled as
information measures in the literature.
This is encapsulated by the supertype [`InformationMeasure`](@ref), whose subtypes are known (e.g., [`Shannon`](@ref)) and less-known (e.g., [`Curado`](@ref)) definitions of information measures, the most common of which are entropies.

But even "entropy" is an umbrella term that may mean several computationally, and sometimes even fundamentally, different quantities. In ComplexityMeasures.jl, we provide the generic function [`information`](@ref) that tries to both clarify disparate information-related measures, while unifying them under a common interface that highlights the modular nature of the term "information measure".

An information measure as defined by a subtype of [`InformationMeasure`](@ref). However, estimating an information measure can be separated, on the highest level, into two main types:

1. **Discrete** information measures are functions of [probability mass functions](https://en.wikipedia.org/wiki/Probability_mass_function). Computing a discrete information measure boils
    down to two simple steps: defining an outcome space (using [`OutcomeSpace`](@ref)), then
     estimating a probability distribution (using a [`ProbabilitiesEstimator`](@ref)), then
     plugging the estimated probabilities into a discrete estimator of the information measure definition ([`DiscreteInfoEstimator`](@ref)).
2. **Differential or continuous** information measures are functions of
    [probability density functions](https://en.wikipedia.org/wiki/Probability_density_function),
    which are *integrals*. Computing differential information measures therefore rely on estimating
    some density functional. For this task, we provide [`DifferentialInfoEstimator`](@ref)s,
    which compute information measures via alternate means, without explicitly computing some
    probability distribution. For example, the [`Correa`](@ref) estimator computes the
    Shannon differential entropy using order statistics.

Crucially, many quantities in the nonlinear dynamics literature that are named as
entropies, such as "permutation entropy" ([`entropy_permutation`](@ref)) and
"wavelet entropy" ([`entropy_wavelet`](@ref)), are *not really new entropies*.
They are the good old discrete Shannon entropy ([`Shannon`](@ref)), but calculated with *new probabilities estimators*. In turn, [`Shannon`](@ref) entropy is just one of many entropies and entropies are a subset of information measures.
Nevertheless, we acknolwedge that names such as "permutation entropy" are commonplace, so in ComplexityMeasures.jl we provide convenience functions like [`entropy_permutation`](@ref).
However, we emphasize that these functions really aren't anything more than
2-lines-of-code wrappers that call [`information`](@ref) with the appropriate
[`ProbabilitiesEstimator`](@ref) and [`InformationMeasure`](@ref).

### Other complexity measures

Other complexity measures which are not functionals of probability mass or density functions, yet still output some quantity related with complexity analysis, are called **complexity measures** within ComplexityMeasures.jl. They can be found in [Complexity measures](@ref) page.
This includes measures like sample entropy and approximate entropy (which, even if named entropies, are not entropies in the formal mathematical sense).

A complexity measure is encapsulated by a [`ComplexityEstimator`](@ref) that can be given to the [`complexity`](@ref) function to obtain the corresponding numerical value.
We stress that the separation between **information measure** and **complexity measure** is purely pragmatic, to establish a generic and extendable software interface within ComplexityMeasures.jl. One can always argue that _"measure X should have been complexity and/or information"_ but in most cases this separation does not matter, it is only important that a classification choice is done and we stick with it.

## [Input data for ComplexityMeasures.jl](@id input_data)

The input data type typically depend on the probability estimator chosen.
In general though, the standard DynamicalSystems.jl approach is taken and as such we have three types of input data:

- *Timeseries*, which are `AbstractVector{<:Real}`, used in e.g. with [`WaveletOverlap`](@ref).
- *Multi-variate timeseries, or datasets, or state space sets*, which are [`StateSpaceSet`](@ref)s, used e.g. with [`NaiveKernel`](@ref). The short syntax `SSSet` may be used instead of `StateSpaceSet`.
- *Spatial data*, which are higher dimensional standard `Array`s, used e.g. with  [`SpatialSymbolicPermutation`](@ref).

```@docs
StateSpaceSet
```
