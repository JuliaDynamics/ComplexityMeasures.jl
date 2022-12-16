# Multiscale

## Multiscale API

The multiscale API is defined by the functions

- [`multiscale`](@ref)
- [`multiscale_normalized`](@ref)
- [`downsample`](@ref)

which dispatch any of the [`MultiScaleAlgorithm`](@ref)s listed below.

```@docs
MultiScaleAlgorithm
Regular
Composite
```

## Multiscale entropy

```@docs
multiscale
multiscale_normalized
downsample
```

## Available literature methods

A non-exhaustive list of literature methods, and the syntax to compute them, are listed
below. Please open an issue or make a pull-request to
[Entropies.jl](https://github.com/JuliaDynamics/Entropies.jl) if you find a literature
method missing from this list, or if you publish a paper based on some new multiscale
combination.

| Method                                          | Syntax                                                             | Reference                       |
| ----------------------------------------------- | ------------------------------------------------------------------ | ------------------------------- |
| Refined composite multiscale dispersion entropy | `multiscale(Composite(), Dispersion(), est, x, normalized = true)` | Azami et al. (2017)[^Azami2017] |

[^Azami2017]:
    Azami, H., Rostaghi, M., Abásolo, D., & Escudero, J. (2017). Refined
    composite multiscale dispersion entropy and its application to biomedical signals.
    IEEE Transactions on Biomedical Engineering, 64(12), 2872-2879.