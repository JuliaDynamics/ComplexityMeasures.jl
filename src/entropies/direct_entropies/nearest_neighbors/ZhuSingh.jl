using DelayEmbeddings: minmaxima
using SpecialFunctions: digamma
using Entropies: Entropy, IndirectEntropy
using Neighborhood: KDTree, Chebyshev, bulkisearch, Theiler, NeighborNumber

export ZhuSingh

"""
    ZhuSingh <: IndirectEntropy
    ZhuSingh(k = 1, w = 0, base = MathConstants.e)

The `ZhuSingh` indirect entropy estimator (Zhu et al., 2015)[^Zhu2015] estimates the Shannon
entropy of `x` (a multi-dimensional `Dataset`) to the given `base`.

Like [`Zhu`](@ref), this estimator approximates probabilities within hyperrectangles
surrounding each point `xᵢ ∈ x` using using `k` nearest neighbor searches. However,
it also considers the number of neighbors falling on the borders of these hyperrectangles.
This estimator is an extension to the entropy estimator in Singh et al. (2003).

`w` is the Theiler window, which determines if temporal neighbors are excluded
during neighbor searches (defaults to `0`, meaning that only the point itself is excluded
when searching for neighbours).

[^Zhu2015]:
    Zhu, J., Bellanger, J. J., Shu, H., & Le Bouquin Jeannès, R. (2015). Contribution to
    transfer entropy estimation via the k-nearest-neighbors approach. Entropy, 17(6),
    4173-4201.
[^Singh2003]:
    Singh, H., Misra, N., Hnizdo, V., Fedorowicz, A., & Demchuk, E. (2003). Nearest
    neighbor estimates of entropy. American journal of mathematical and management
    sciences, 23(3-4), 301-321.
"""
Base.@kwdef struct ZhuSingh{B} <: IndirectEntropy
    k::Int = 1
    w::Int = 0
    base::B = MathConstants.e

    function ZhuSingh(k::Int, w::Int, base::B) where B
        new{B}(k, w, base)
    end
end

"""
    n_borderpoints(xᵢ, nns, dists) → ξ

Compute `ξ`, which is how many of `xᵢ`'s neighbor points `xⱼ ∈ nns` fall on the border of
the minimal-volume rectangle with `xᵢ` at its center.

`dists[k]` should be the maximum distance from `xᵢ[k]` to any other point along the k-th
dimension, and `length(dists)` is the total dimension.
"""
n_borderpoints(xᵢ, nns, dists) = count(any(abs.(xᵢ .- xⱼ) .== dists) for xⱼ in nns)

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

function entropy(e::ZhuSingh, x::AbstractDataset{D, T}) where {D, T}
    (; k, w, base) = e
    N = length(x)
    tree = KDTree(x, Euclidean())
    nn_idxs = bulkisearch(tree, x, NeighborNumber(k), Theiler(w))

    mean_logvol, mean_digammaξ = mean_logvolumes_and_digamma(x, nn_idxs, N, k)
    h = digamma(N) + mean_logvol - mean_digammaξ

    return h / log(base, MathConstants.e)
end
