using StateSpaceSets: AbstractStateSpaceSet
using Neighborhood: Euclidean, KDTree, NeighborNumber, Theiler
using Neighborhood: bulksearch
using SpecialFunctions: digamma

export Gao

"""
    Gao <: DifferentialInfoEstimator
    Gao(definition = Shannon(); k = 1, w = 0, corrected = true)

The `Gao` estimator [Gao2015](@cite) computes the [`Shannon`](@ref)
differential [`information`](@ref), using a `k`-th nearest-neighbor approach
based on [Singh2003](@citet), with logarithms to the `base` specified in
`definition`.

`w` is the Theiler window, which determines if temporal neighbors are excluded
during neighbor searches (defaults to `0`, meaning that only the point itself is excluded
when searching for neighbours).

[Gao2015](@citet) give two variants of this estimator. If `corrected == false`, then the
uncorrected version is used. If `corrected == true`, then the corrected version is used,
which ensures that the estimator is asymptotically unbiased.

## Description

Assume we have samples ``\\{\\bf{x}_1, \\bf{x}_2, \\ldots, \\bf{x}_N \\}`` from a
continuous random variable ``X \\in \\mathbb{R}^d`` with support ``\\mathcal{X}`` and
density function``f : \\mathbb{R}^d \\to \\mathbb{R}``. `KozachenkoLeonenko` estimates
the [`Shannon`](@ref) differential entropy

```math
H(X) = \\int_{\\mathcal{X}} f(x) \\log f(x) dx = \\mathbb{E}[-\\log(f(X))].
```
"""
struct Gao{I <: InformationMeasure} <: NNDifferentialInfoEstimator{I}
    definition::I
    k::Int
    w::Int
    corrected::Bool
end
function Gao(definition = Shannon(); k = 1, w = 0, corrected = true)
    return Gao(definition, k, w, corrected)
end

function information(est::Gao{<:Shannon}, x::AbstractStateSpaceSet{D}) where D
    (; k, w) = est
    N = length(x)
    f = (k  * gamma(D / 2 + 1)) / ( (N - 1) * π^(D / 2))
    tree = KDTree(x, Euclidean())
    idxs, ds = bulksearch(tree, x, NeighborNumber(k), Theiler(w))

    # The estimated entropy has "unit" [nats]
    h = -(1 / N) * sum(log(f * 1 / last(dᵢ)^D) for dᵢ in ds)
    if est.corrected
        correction = digamma(k) - log(k)
        h -= correction
    end

    return convert_logunit(h, ℯ, est.definition.base) # convert to target unit *after* correction
end
