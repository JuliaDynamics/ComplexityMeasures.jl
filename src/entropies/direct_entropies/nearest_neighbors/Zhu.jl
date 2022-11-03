export Zhu

"""
    Zhu <: IndirectEntropy
    Zhu(k = 1, w = 1, base = MathConstants.e)

The `Zhu` indirect entropy estimator (Zhu et al., 2015)[^Zhu2015] estimates the Shannon
entropy of `x` (a multi-dimensional `Dataset`) to the given `base`, by approximating
probabilities within hyperrectangles surrounding each point `xᵢ ∈ x` using
using `k` nearest neighbor searches.

This estimator is an extension to [`KozachenkoLeonenko`](@ref).

`w` is the Theiler window, which determines if temporal neighbors are excluded
during neighbor searches (defaults to `0`, meaning that only the point itself is excluded
when searching for neighbours).

[^Zhu2015]:
    Zhu, J., Bellanger, J. J., Shu, H., & Le Bouquin Jeannès, R. (2015). Contribution to
    transfer entropy estimation via the k-nearest-neighbors approach. Entropy, 17(6),
    4173-4201.
"""
Base.@kwdef struct Zhu{B} <: IndirectEntropy
    k::Int = 1
    w::Int = 1
    base::B = MathConstants.e
end

function entropy(e::Zhu, x::AbstractDataset{D, T}) where {D, T}
    (; k, w, base) = e
    N = length(x)
    tree = KDTree(x, Euclidean())
    nn_idxs = bulkisearch(tree, x, NeighborNumber(k), Theiler(w))
    h = digamma(N) + mean_logvolumes(x, nn_idxs, N) - digamma(k) + (D - 1) / k
    return h / log(base, MathConstants.e)
end

function mean_logvolumes(x, nn_idxs, N::Int)
    v = 0.0
    for (i, (xᵢ, nn_idxsᵢ)) in enumerate(zip(x, nn_idxs))
        nnsᵢ = @views x[nn_idxsᵢ] # the actual coordinates of the points
        v += log(MathConstants.e, volume_minirect(xᵢ, nnsᵢ))
    end
    return v / N
end

"""
    volume_minirect(xᵢ, nns)

Compute the volume of the minimal enclosing rectangle with `xᵢ` at its center and
containing all points `nᵢ ∈ nns` either within the rectangle or on one of its borders.

This function respects the coordinate system of the input data, i.e. it does not perform
any rotation (which would be computationally more demanding because we'd need to find the
convex hull of `nns`, but could potentially give more accurate results).
"""
function volume_minirect(xᵢ, nns::AbstractDataset)
    mini, maxi = minmaxima(nns)
    # dists[d] is the maximum distance from the point xᵢ to any neighbor in dimension d
    # We take twice those distances to ensure xᵢ is at the centre of the rectangle.
    dists = max.(maxi .- xᵢ, xᵢ .- mini)
    return prod(dists .* 2)
end
