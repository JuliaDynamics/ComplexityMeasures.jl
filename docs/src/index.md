# Entropies.jl

This package provides estimators for probabilities, entropies, and complexity measures for timeseries, nonlinear dynamics and complex systems. It is used in the [CausalityTools.jl](https://github.com/JuliaDynamics/CausalityTools.jl) and [DynamicalSystems.jl](https://github.com/JuliaDynamics/DynamicalSystems.jl) packages.

## Input data

This package assumes that your data is represented by the `Dataset`-type from [`DelayEmbeddings.jl`](https://github.com/JuliaDynamics/DelayEmbeddings.jl), where each observation is a D-dimensional data point. See the [`DynamicalSystems.jl` documentation](https://juliadynamics.github.io/DynamicalSystems.jl/dev/) for more info. Univariate timeseries given as
`AbstractVector{<:Real}` also work with some estimators, but are treated differently
based on which method for probability/entropy estimation is applied.

## API

The main **API** of this package is contained in three functions:

* [`probabilities`](@ref) which computes probability distributions of given datasets
* [`entropy_renyi`](@ref) which uses the output of [`probabilities`](@ref), or a set of
    pre-computed [`Probabilities`](@ref), to calculate entropies.
* [`entropy_tsallis`](@ref) which uses the output of [`probabilities`](@ref), or a set of
    pre-computed [`Probabilities`](@ref), to calculate Tsallis entropies.

These functions dispatch estimators listed [here](@ref estimators).
