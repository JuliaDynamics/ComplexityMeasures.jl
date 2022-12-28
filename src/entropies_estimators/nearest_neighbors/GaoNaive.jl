using StateSpaceSets: AbstractDataset
using Neighborhood: Euclidean, KDTree, NeighborNumber, Theiler
using Neighborhood: bulksearch
using SpecialFunctions: digamma

export GaoNaive, GaoNaiveCorrected

"""
    GaoNaive <: DifferentialEntropyEstimator
    GaoNaive(; k = 1, w = 0)

The `GaoNaive` estimator (Gao et al., 2015) computes the [`Shannon`](@ref)
differential [`entropy`](@ref), using a `k`-th nearest-neighbor approach
based on Singh et al. (2003)[^Singh2003].

`w` is the Theiler window, which determines if temporal neighbors are excluded
during neighbor searches (defaults to `0`, meaning that only the point itself is excluded
when searching for neighbours).

## Description

Assume we have samples ``\\{\\bf{x}_1, \\bf{x}_2, \\ldots, \\bf{x}_N \\}`` from a
continuous random variable ``X \\in \\mathbb{R}^d`` with support ``\\mathcal{X}`` and
density function``f : \\mathbb{R}^d \\to \\mathbb{R}``. `KozachenkoLeonenko` estimates
the [Shannon](@ref) differential entropy

```math
H(X) = \\int_{\\mathcal{X}} f(x) \\log f(x) dx = \\mathbb{E}[-\\log(f(X))]
```

[^Gao2005]:
    Gao, S., Ver Steeg, G., & Galstyan, A. (2015, February). Efficient estimation of
    mutual information for strongly dependent variables. In Artificial intelligence and
        statistics (pp. 277-286). PMLR.
[^Singh2003]:
    Singh, H., Misra, N., Hnizdo, V., Fedorowicz, A., & Demchuk, E. (2003). Nearest
    neighbor estimates of entropy. American journal of mathematical and management
    sciences, 23(3-4), 301-321.
"""
Base.@kwdef struct GaoNaive <: DifferentialEntropyEstimator
    k::Int = 1
    w::Int = 0
end

function entropy(e::Shannon, est::GaoNaive, x::AbstractDataset{D}) where D
    (; k, w) = est
    N = length(x)
    f = (k  * gamma(D / 2 + 1)) / ( (N - 1) * π^(D / 2))
    tree = KDTree(x, Euclidean())
    idxs, ds = bulksearch(tree, x, NeighborNumber(k), Theiler(w))
    h = -(1 / N) * sum(log(f * 1 / last(dᵢ)^D) for dᵢ in ds) # in nats
    return h / log(e.base, ℯ) # convert to target unit
end

"""
    GaoNaiveCorrected <: DifferentialEntropyEstimator
    GaoNaiveCorrected(; k = 1, w = 0)

The `GaoNaiveCorrected` estimator (Gao et al., 2015), computes the [`Shannon`](@ref)
differential [`entropy`](@ref), using a `k`-th nearest-neighbor approach based on
Singh et al. (2003)[^Singh2003].

Keyword arguments are the same as for [`GaoNaive`](@ref).

## Description

`GaoNaiveCorrected` is identical to the [`GaoNaive`](@ref) estimator,
except it adds correction terms that ensures the estimator is asymptotically unbiased.

[^Gao2005]:
    Gao, S., Ver Steeg, G., & Galstyan, A. (2015, February). Efficient estimation of
    mutual information for strongly dependent variables. In Artificial intelligence and
        statistics (pp. 277-286). PMLR.
[^Singh2003]:
    Singh, H., Misra, N., Hnizdo, V., Fedorowicz, A., & Demchuk, E. (2003). Nearest
    neighbor estimates of entropy. American journal of mathematical and management
    sciences, 23(3-4), 301-321.
"""
Base.@kwdef struct GaoNaiveCorrected <: DifferentialEntropyEstimator
    k::Int = 1
    w::Int = 0
end

function entropy(e::Shannon, est::GaoNaiveCorrected, x::AbstractDataset{D}) where D
    (; k, w) = est
    N = length(x)
    f = (k  * gamma(D / 2 + 1)) / ( (N - 1) * π^(D / 2))
    tree = KDTree(x, Euclidean())
    idxs, ds = bulksearch(tree, x, NeighborNumber(k), Theiler(w))
    corr = digamma(k) - log(k)
    h = -(1 / N) * sum(log(f * 1 / last(dᵢ)^D) for dᵢ in ds) - corr # in nats
    return h / log(e.base, ℯ) # convert to target unit after correction
end
