# ComplexityMeasures.jl

```@docs
ComplexityMeasures
```

## [Content and terminology](@id terminology)

!!! note
    The documentation here follows (loosely) chapter 5 of
    [Nonlinear Dynamics](https://link.springer.com/book/10.1007/978-3-030-91032-7),
    Datseris & Parlitz, Springer 2022.

Before exploring the features of ComplexityMeasures.jl, it is useful to read through this terminology section. Here, we briefly review important complexity-related concepts and names from the scientific literature, and outline how we've structured ComplexityMeasures.jl around these concepts.

In these scientific literature, words like *probabilities*, *entropies*, and other *complexity measures* are used (and abused) in multiple contexts, and are often used interchangeably to describe similar concepts. The API and documentation of ComplexityMeasures.jl aim to clarify the meaning and usage of these words, and to provide simple ways to obtain probabilities, entropies, or other complexity measures
from input data.

For ComplexityMeasures.jl _entropies are also complexity measures_, while sometimes a distinction is made so that "complexity measures" means anything beyond entropy. However we believe the general nonlinear dynamics community agrees with our take, as most papers that introduce different entropy flavors, call them complexity measures. Example: _"Permutation Entropy: A Natural Complexity Measure for Time Series"_ from Brandt and Pompe 2002.

### Probabilities

Entropies and other complexity measures are typically computed based on _probability
distributions_,
which we simply refer to as "probabilities".
Probabilities can be obtained from input data in a plethora of different ways.
The central API function that returns a probability distribution
is [`probabilities`](@ref), which takes in a subtype of [`ProbabilitiesEstimator`](@ref)
to specify how the probabilities are computed.
All available estimators can be found in the [estimators page](@ref probabilities_estimators).

### Entropies

Entropy is an established concept in statistics, information theory, and nonlinear dynamics.
However, it is also an umbrella term that may mean several computationally, and sometimes
even fundamentally, different quantities.
In ComplexityMeasures.jl, we provide the generic
function [`entropy`](@ref) that tries to both clarify disparate entropy concepts, while
unifying them under a common interface that highlights the modular nature of the word
"entropy".

On the highest level, there are two main types of entropy.

- *Discrete* entropies are functions of [probability mass functions](https://en.wikipedia.org/wiki/Probability_mass_function). Computing a discrete entropy boils
    down to two simple steps: first estimating a probability distribution, then plugging
    the estimated probabilities into an estimator of a so-called "generalized entropy" definition.
    Internally, this is literally just a few lines of code where we first apply some
    [`ProbabilitiesEstimator`](@ref) to the input data, and feed the resulting
    [`probabilities`](@ref) to [`entropy`](@ref) with some [`DiscreteEntropyEstimator`](@ref).
- *Differential/continuous* entropies are functions of
    [probability density functions](https://en.wikipedia.org/wiki/Probability_density_function),
    which are *integrals*. Computing differential entropies therefore rely on estimating
    some density functional. For this task, we provide [`DifferentialEntropyEstimator`](@ref)s,
    which compute entropies via alternate means, without explicitly computing some
    probability distribution. For example, the [`Correa`](@ref) estimator computes the
    Shannon differential entropy using order statistics.

Crucially, many quantities in the nonlinear dynamics literature that are named as
entropies, such as "permutation entropy" ([`entropy_permutation`](@ref)) and
"wavelet entropy" ([`entropy_wavelet`](@ref)), are *not really new entropies*.
They are the good old discrete Shannon entropy ([`Shannon`](@ref)), but calculated with
*new probabilities estimators*.

Even though the names of these methods (e.g. "wavelet entropy") sound like names for new
entropies, what they actually do is to devise novel
ways of calculating probabilities from data, and then plug those probabilities into formal
discrete entropy formulas such as
the Shannon entropy. These probabilities estimators are of course smartly created so that
they elegantly highlight important complexity-related aspects of the data.

Names for methods such as "permutation entropy" are commonplace, so in
ComplexityMeasures.jl we provide convenience functions like [`entropy_permutation`](@ref).
However, we emphasize that these functions really aren't anything more than
2-lines-of-code wrappers that call [`entropy`](@ref) with the appropriate
[`ProbabilitiesEstimator`](@ref).

What are *genuinely different entropies* are different definitions of entropy. And there
are a lot of them! Examples are [`Shannon`](@ref) (the classic), [`Renyi`](@ref) or
[`Tsallis`](@ref) entropy. These different definitions can be found in
[`EntropyDefinition`](@ref).

### Other complexity measures

Other complexity measures, which strictly speaking don't compute entropies, and may or may not explicitly compute probability distributions, are found in
[Complexity measures](@ref) page.
This includes measures like sample entropy and approximate entropy.

## [Input data for ComplexityMeasures.jl](@id input_data)

The input data type typically depend on the probability estimator chosen.
In general though, the standard DynamicalSystems.jl approach is taken and as such we have three types of input data:

- *Timeseries*, which are `AbstractVector{<:Real}`, used in e.g. with [`WaveletOverlap`](@ref).
- *Multi-variate timeseries, or datasets, or state space sets*, which are [`Dataset`](@ref)s, used e.g. with [`NaiveKernel`](@ref).
- *Spatial data*, which are higher dimensional standard `Array`s, used e.g. with  [`SpatialSymbolicPermutation`](@ref).

```@docs
Dataset
```
