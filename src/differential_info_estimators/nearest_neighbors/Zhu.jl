export Zhu

"""
    Zhu <: DifferentialInfoEstimator
    Zhu(; definition = Shannon(), k = 1, w = 0)

The `Zhu` estimator [Zhu2015](@cite) is an extension to
[`KozachenkoLeonenko`](@ref), and computes the [`Shannon`](@ref)
differential [`information`](@ref) of a multi-dimensional [`StateSpaceSet`](@ref),
with logarithms to the `base` specified in `definition`.

## Description

Assume we have samples ``\\{\\bf{x}_1, \\bf{x}_2, \\ldots, \\bf{x}_N \\}`` from a
continuous random variable ``X \\in \\mathbb{R}^d`` with support ``\\mathcal{X}`` and
density function``f : \\mathbb{R}^d \\to \\mathbb{R}``. `Zhu` estimates the [`Shannon`](@ref)
differential entropy

```math
H(X) = \\int_{\\mathcal{X}} f(x) \\log f(x) dx = \\mathbb{E}[-\\log(f(X))]
```

by approximating densities within hyperrectangles surrounding each point `xᵢ ∈ x` using
using `k` nearest neighbor searches. `w` is the Theiler window, which determines if
temporal neighbors are excluded during neighbor searches (defaults to `0`, meaning that
only the point itself is excluded when searching for neighbours).

See also: [`information`](@ref), [`KozachenkoLeonenko`](@ref),
[`DifferentialInfoEstimator`](@ref).
"""
struct Zhu{I <: InformationMeasure} <: NNDifferentialInfoEstimator{I}
    definition::I
    k::Int
    w::Int
end
function Zhu(definition = Shannon(); k = 1, w = 0)
    return Zhu(definition, k, w)
end

function information(est::Zhu{<:Shannon}, x::AbstractStateSpaceSet{D, T}) where {D, T}
    (; k, w) = est
    N = length(x)
    tree = KDTree(x, Euclidean())
    nn_idxs = bulkisearch(tree, x, NeighborNumber(k), Theiler(w))
    # The estimated entropy has "unit" [nats]
    h = digamma(N) + mean_logvolumes(x, nn_idxs, N) - digamma(k) + (D - 1) / k
    return convert_logunit(h, ℯ, est.definition.base)
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
