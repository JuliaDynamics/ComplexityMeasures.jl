using DelayEmbeddings: minmaxima
using SpecialFunctions: digamma
using ComplexityMeasures: InformationMeasure, DifferentialInfoEstimator
using Neighborhood: KDTree, Chebyshev, bulkisearch, Theiler, NeighborNumber

export ZhuSingh

"""
    ZhuSingh <: DifferentialInfoEstimator
    ZhuSingh(definition = Shannon(); k = 1, w = 0)

The `ZhuSingh` estimator
(Zhu et al., 2015; Singh et al., 2003)[Zhu2015](@cite)[Singh2003](@cite) computes the
[`Shannon`](@ref) differential [`information`](@ref) of a multi-dimensional
[`StateSpaceSet`](@ref), with logarithms to the `base` specified in `definition`.

## Description

Assume we have samples ``\\{\\bf{x}_1, \\bf{x}_2, \\ldots, \\bf{x}_N \\}`` from a
continuous random variable ``X \\in \\mathbb{R}^d`` with support ``\\mathcal{X}`` and
density function``f : \\mathbb{R}^d \\to \\mathbb{R}``. `ZhuSingh` estimates the
[Shannon](@ref) differential entropy

```math
H(X) = \\int_{\\mathcal{X}} f(x) \\log f(x) dx = \\mathbb{E}[-\\log(f(X))].
```

Like [`Zhu`](@ref), this estimator approximates probabilities within hyperrectangles
surrounding each point `xᵢ ∈ x` using using `k` nearest neighbor searches. However,
it also considers the number of neighbors falling on the borders of these hyperrectangles.
This estimator is an extension to the entropy estimator in Singh et al. (2003).

`w` is the Theiler window, which determines if temporal neighbors are excluded
during neighbor searches (defaults to `0`, meaning that only the point itself is excluded
when searching for neighbours).

See also: [`information`](@ref), [`DifferentialInfoEstimator`](@ref).
"""
struct ZhuSingh{I <: InformationMeasure} <: NNDifferentialInfoEstimator{I}
    definition::I
    k::Int
    w::Int
end
function ZhuSingh(definition = Shannon(); k = 1, w = 0)
    return ZhuSingh(definition, k, w)
end

function information(est::ZhuSingh{<:Shannon}, x::AbstractStateSpaceSet{D, T}) where {D, T}
    (; k, w) = est
    N = length(x)
    tree = KDTree(x, Euclidean())
    nn_idxs = bulkisearch(tree, x, NeighborNumber(k), Theiler(w))
    mean_logvol, mean_digammaξ = mean_logvolumes_and_digamma(x, nn_idxs, N, k)
    # The estimated entropy has "unit" [nats]
    h = digamma(N) + mean_logvol - mean_digammaξ
    return convert_logunit(h, ℯ, est.definition.base)
end

function mean_logvolumes_and_digamma(x, nn_idxs, N::Int, k::Int)
    T = eltype(0.0)
    logvol::T = 0.0
    digammaξ::T = 0.0
    for (i, (xᵢ, nn_idxsᵢ)) in enumerate(zip(x, nn_idxs))
        nnsᵢ = @views x[nn_idxsᵢ] # the actual coordinates of the points
        distsᵢ = maxdists(xᵢ, nnsᵢ)
        ξ = n_borderpoints(xᵢ, nnsᵢ, distsᵢ)
        digammaξ += digamma(k - ξ + 1)
        logvol += log(MathConstants.e, volume_minimal_rect(distsᵢ))
    end
    logvol /= N
    digammaξ /= N

    return logvol, digammaξ
end

"""
    volume_minimal_rect(dists) → vol

Compute the volume of a (hyper)-rectangle where the distance from its centre along the
`k`-th dimension is given by `dists[k]`, and `length(dists)` is the total dimension.
"""
volume_minimal_rect(dists) = prod(dists .* 2)

"""
    n_borderpoints(xᵢ, nns, dists) → ξ

Compute `ξ`, which is how many of `xᵢ`'s neighbor points `xⱼ ∈ nns` fall on the border of
the minimal-volume rectangle with `xᵢ` at its center.

`dists[k]` should be the maximum distance from `xᵢ[k]` to any other point along the k-th
dimension, and `length(dists)` is the total dimension.
"""
n_borderpoints(xᵢ, nns, dists) = count(any(abs.(xᵢ .- xⱼ) .== dists) for xⱼ in nns)
