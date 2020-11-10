# Entropies.jl

This package provides probability/entropy estimators used for entropy computations in the [CausalityTools.jl](https://github.com/JuliaDynamics/CausalityTools.jl) and [DynamicalSystems.jl](https://github.com/JuliaDynamics/DynamicalSystems.jl) packages.

Most of the code in this package assumes that your data is represented by the `Dataset`-type from [`DelayEmbeddings.jl`](https://github.com/JuliaDynamics/DelayEmbeddings.jl), where each observation is a D-dimensional data point represented by a static vector. See the [`DynamicalSystems.jl` documentation](https://juliadynamics.github.io/DynamicalSystems.jl/dev/) for more info. Univariate time series given as 
`AbstractVector{<:Real}` also work with some estimators, but are treated differently 
based on which method for probability/entropy estimation is applied.

## Generalized entropy

Generalized entropy is a property of probability distributions. In Entropies.jl, the 
generalized entropy can be estimated 

- directly on pre-computed [`Probabilites`](@ref),
- directly using some `EntropyEstimator`, or
- indirectly by using some `ProbabilityEstimator` to estimate a probability distribution, 
    then computing entropy on that distribution.

Check the docstrings for individual estimators to see which methods work on which kinds of data.

```@docs
Entropies.genentropy(p::Probabilities)
```

## Probabilities

```@docs
Probabilities
probabilities
probabilities!
```
