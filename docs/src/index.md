# Entropies.jl

This package provides estimators for probabilities, entropies, and complexity measures for timeseries, nonlinear dynamics and complex systems. It is used in the [CausalityTools.jl](https://github.com/JuliaDynamics/CausalityTools.jl) and [DynamicalSystems.jl](https://github.com/JuliaDynamics/DynamicalSystems.jl) packages.

This package assumes that your data is represented by the `Dataset`-type from [`DelayEmbeddings.jl`](https://github.com/JuliaDynamics/DelayEmbeddings.jl), where each observation is a D-dimensional data point. See the [`DynamicalSystems.jl` documentation](https://juliadynamics.github.io/DynamicalSystems.jl/dev/) for more info. Univariate timeseries given as
`AbstractVector{<:Real}` also work with some estimators, but are treated differently
based on which method for probability/entropy estimation is applied.

## API & terminology

In the literature, the term "entropy" is used (and abused) in multiple contexts. We use the following distinctions.

- [Generalized Rényi and Tsallis entropies](@ref generalized_entropies) are theoretically well-founded concepts that are functions of *probability distributions*.

- [Shannon entropy](@ref shannon_entropies) is a special case of Rényi and Tsallis entropies. We provide convenience functions for most common Shannon entropy estimators.

- [*Probability estimation*](@ref estimators) is a separate but required step to compute entropies. We provide a range of probability estimators. These estimators can be used in isolation for estimating probability distributions, or for computing generalized Rényi and Tsallis entropies.

- "Entropy-like" complexity measures, which strictly speaking don't compute entropies, and may or may not explicitly compute probability distributions, appear in the [Complexity measures](@ref) section.

The main **API** of this package is thus contained in three functions:

- [`probabilities`](@ref), which computes probability distributions of given datasets.
- [`entropy_renyi`](@ref), which uses the output of [`probabilities`](@ref), or a set of pre-computed [`Probabilities`](@ref), to calculate entropies.
- [`entropy_tsallis`](@ref), which uses the output of [`probabilities`](@ref), or a set of pre-computed [`Probabilities`](@ref), to calculate Tsallis entropies.
- Convenience functions for commonly used methods appear throughout the documentation.

These functions dispatch on the probability estimators listed [here](@ref estimators).

*Note: there are fewer probability estimators than there are Shannon entropy estimators, because some Shannon entropy are indirect, in the sense that they don't explicitly compute probability distributions*
