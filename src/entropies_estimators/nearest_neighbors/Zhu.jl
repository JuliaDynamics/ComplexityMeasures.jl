export Zhu

"""
    Zhu <: DiffEntropyEst
    Zhu(; k = 1, w = 0, base = 2)

The `Zhu` estimator (Zhu et al., 2015)[^Zhu2015] is an extension to
[`KozachenkoLeonenko`](@ref), and computes the [`Shannon`](@ref)
differential [`entropy`](@ref) of a multi-dimensional [`StateSpaceSet`](@ref) in the given `base`.

## Description

Assume we have samples ``\\{\\bf{x}_1, \\bf{x}_2, \\ldots, \\bf{x}_N \\}`` from a
continuous random variable ``X \\in \\mathbb{R}^d`` with support ``\\mathcal{X}`` and
density function``f : \\mathbb{R}^d \\to \\mathbb{R}``. `Zhu` estimates the [Shannon](@ref)
differential entropy

```math
H(X) = \\int_{\\mathcal{X}} f(x) \\log f(x) dx = \\mathbb{E}[-\\log(f(X))]
```

by approximating densities within hyperrectangles surrounding each point `xᵢ ∈ x` using
using `k` nearest neighbor searches. `w` is the Theiler window, which determines if
temporal neighbors are excluded during neighbor searches (defaults to `0`, meaning that
only the point itself is excluded when searching for neighbours).

See also: [`entropy`](@ref), [`KozachenkoLeonenko`](@ref), [`DifferentialEntropyEstimator`](@ref).

[^Zhu2015]:
    Zhu, J., Bellanger, J. J., Shu, H., & Le Bouquin Jeannès, R. (2015). Contribution to
    transfer entropy estimation via the k-nearest-neighbors approach. EntropyDefinition, 17(6),
    4173-4201.
"""
Base.@kwdef struct Zhu{B} <: NNDiffEntropyEst
    k::Int = 1
    w::Int = 0
    base::B = 2
end

function entropy(est::Zhu, x::AbstractStateSpaceSet{D, T}) where {D, T}
    (; k, w) = est
    N = length(x)
    tree = KDTree(x, Euclidean())
    nn_idxs = bulkisearch(tree, x, NeighborNumber(k), Theiler(w))
    h = digamma(N) + mean_logvolumes(x, nn_idxs, N) - digamma(k) + (D - 1) / k
    return h / log(est.base, MathConstants.e)
end

function mean_logvolumes(x, nn_idxs, N::Int)
    v = 0.0
    for (xᵢ, nn_idxsᵢ) in zip(x, nn_idxs)
        nnsᵢ = @views x[nn_idxsᵢ] # the actual coordinates of the points
        v += log(MathConstants.e, volume_minimal_rect(xᵢ, nnsᵢ))
    end
    return v / N
end

"""
    volume_minimal_rect(xᵢ, nns) → vol

Compute the volume of the minimal enclosing rectangle with `xᵢ` at its center and
containing all points `nᵢ ∈ nns` either within the rectangle or on one of its borders.

This function respects the coordinate system of the input data, i.e. it does not perform
any rotation (which would be computationally more demanding because we'd need to find the
convex hull of `nns`, but could potentially give more accurate results).
"""
volume_minimal_rect(xᵢ, nns::AbstractStateSpaceSet) = prod(maxdists(xᵢ, nns) .* 2)

"""
    maxdists(xᵢ, nns) → dists

Compute the maximum distance from `xᵢ` to the points `xⱼ ∈ nns` along each dimension,
i.e. `dists[k] = max{xᵢ[k], xⱼ[k]}` for `j = 1, 2, ..., length(x)`.
"""
function maxdists(xᵢ, nns)
    mini, maxi = minmaxima(nns)
    return max.(maxi .- xᵢ, xᵢ .- mini)
end
