# Entropies.jl

This package provides entropy estimators used for entropy computations in the [CausalityTools.jl](https://github.com/JuliaDynamics/CausalityTools.jl) and [DynamicalSystems.jl](https://github.com/JuliaDynamics/DynamicalSystems.jl) packages.

Most of the code in this package assumes that your data is represented by the `Dataset`-type from [`DelayEmbeddings.jl`](https://github.com/JuliaDynamics/DelayEmbeddings.jl), where each observation is a D-dimensional data point represented by a static vector. See the [`DynamicalSystems.jl` documentation](https://juliadynamics.github.io/DynamicalSystems.jl/dev/) for more info.