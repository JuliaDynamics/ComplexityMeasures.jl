using StateSpaceSets: AbstractDataset, Dataset
using Neighborhood: KDTree, NeighborNumber, Theiler
using Neighborhood: bulksearch
using SpecialFunctions: digamma

export Goria

"""
    Goria <: DifferentialEntropyEstimator
    Goria(; k = 1, w = 0)

The `Goria` estimator computes the [`Shannon`](@ref) differential
[`entropy`](@ref) of `x` (a multi-dimensional [`Dataset`](@ref)).

## Description

Assume we have samples ``\\{\\bf{x}_1, \\bf{x}_2, \\ldots, \\bf{x}_N \\}`` from a
continuous random variable ``X \\in \\mathbb{R}^d`` with support ``\\mathcal{X}`` and
density function``f : \\mathbb{R}^d \\to \\mathbb{R}``. `Goria` estimates
the [Shannon](@ref) differential entropy

```math
H(X) = \\int_{\\mathcal{X}} f(x) \\log f(x) dx = \\mathbb{E}[-\\log(f(X))].
```


Specifically, let ``\\bf{n}_1, \\bf{n}_2, \\ldots, \\bf{n}_N`` be the distance of the
samples ``\\{\\bf{x}_1, \\bf{x}_2, \\ldots, \\bf{x}_N \\}`` to their
`k`-th nearest neighbors. Next, let the geometric mean of the distances be

```math
\\hat{\\rho}_k = \\left( \\prod_{i=1}^N \\right)^{\\dfrac{1}{N}}
```
Goria et al. (2005)[^Goria2005]'s estimate of Shannon differential entropy is then

```math
\\hat{H} = m\\hat{\\rho}_k + \\log(N - 1) - \\psi(k) + \\log c_1(m),
```

where ``c_1(m) = \\dfrac{2\\pi^\\frac{m}{2}}{m \\Gamma(m/2)}`` and ``\\psi``
is the digamma function.

[^Goria2005]:
    Goria, M. N., Leonenko, N. N., Mergel, V. V., & Novi Inverardi, P. L. (2005). A new
    class of random vector entropy estimators and its applications in testing statistical
    hypotheses. Journal of Nonparametric Statistics, 17(3), 277-297.
"""
Base.@kwdef struct Goria <: DifferentialEntropyEstimator
    k::Int = 1
    w::Int = 0
end

function entropy(e::Shannon, est::Goria, x::AbstractDataset{D}) where D
    (; k, w) = est
    N = length(x)

    tree = KDTree(x, Euclidean())
    ds = last.(bulksearch(tree, x, NeighborNumber(k), Theiler(w))[2])
    h = D * log(prod(ds .^ (1 / N))) +
          log(N - 1) +
          log(c1(D)) -
          digamma(k)

    return h / log(e.base, ℯ)
end
c1(D::Int) = (2π^(D/2)) / (D* gamma(D/2))
