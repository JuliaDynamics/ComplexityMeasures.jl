# Multiscale

## Multiscale API

The multiscale API is defined by the [multiscale algorithms](@ref multiscale_algorithms)
defined below, and the following functions.

- [`multiscale`](@ref)
- [`multiscale_normalized`](@ref)
- [`downsample`](@ref)

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

### Multiscale complexity measures

| Method  | Syntax | Reference |
| ------------- | ------------- | ------------- |
| Multiscale sample entropy (first moment) | `multiscale(SampleEntropy(), Regular(f = mean), x)` |Costa et al. (2002)[^Costa2002] |
| Generalized multiscale sample entropy (second moment)| `multiscale(SampleEntropy(), Regular(f = std), x)` | Costa et al. (2015)[^Costa2015] |

### Multiscale entropy

| Method  | Syntax | Reference |
| ------------- | ------------- | ------------- |
| Refined composite multiscale dispersion entropy  | `multiscale(Dispersion(), Composite(), x, normalized = true)` | Azami et al. (2017)[^Azami2017] |

## [Algorithms](@id multiscale_algorithms)

```@docs
MultiScaleAlgorithm
Regular
Composite
```

[^Costa2002]:
    Costa, M., Goldberger, A. L., & Peng, C. K. (2002). Multiscale entropy
    analysis of complex physiologic time series. Physical review letters, 89(6), 068102.
[^Costa2015]:
    Costa, M. D., & Goldberger, A. L. (2015). Generalized multiscale entropy
    analysis: Application to quantifying the complex volatility of human heartbeat time
    series. Entropy, 17(3), 1197-1203.
[^Azami2017]:
    Azami, H., Rostaghi, M., Abásolo, D., & Escudero, J. (2017). Refined
    composite multiscale dispersion entropy and its application to biomedical signals.
    IEEE Transactions on Biomedical Engineering, 64(12), 2872-2879.
[^Morabito2012]:
    Morabito, F. C., Labate, D., Foresta, F. L., Bramanti, A., Morabito, G.,
    & Palamara, I. (2012). Multivariate multi-scale permutation entropy for complexity
    analysis of Alzheimer’s disease EEG. Entropy, 14(7), 1186-1202.
