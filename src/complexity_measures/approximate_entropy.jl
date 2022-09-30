using DelayEmbeddings
using Neighborhood
using Distances
using Statistics

export approx_entropy

"""
    approx_entropy(x; m = 2, τ = 1, r = 0.2 * Statistics.std(x), base = MathConstants.e)

Compute the approximate entropy (ApEn; Pincus, 1991)[^Pincus1991] of a univariate
timeseries `x`, using embedding dimension (pattern length) `m`, embedding lag `τ`,
and tolerance radius `r`.

The tolerance radius `r` should be determined from the input data, for example as
some fraction of the standard deviation of the input.

[^Pincus1991]: Pincus, S. M. (1991). Approximate entropy as a measure of system complexity.
    Proceedings of the National Academy of Sciences, 88(6), 2297-2301.
"""
function approx_entropy(x::AbstractVector{T}; m::Int = 2, τ::Int = 1,
        r = 0.2 * std(x), base = MathConstants.e) where T
    m >= 1 || throw(ArgumentError("m must be >= 1. Got m=$(m)."))

    # Definition in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7515030/
    if m == 1
       return compute_ϕ(x, r; m, τ, base)
    else
        ϕᵐ = compute_ϕ(x, r; m, τ, base)
        ϕᵐ⁺¹ = compute_ϕ(x, r; m = m + 1, τ, base)

        return ϕᵐ - ϕᵐ⁺¹
    end
end

approx_entropy(x; kwargs...) =
    throw(ArgumentError(
        "`approx_entropy` not implemented for input data of type $(typeof(x))"
    ))

"""
    compute_ϕ(x, r; m::Int = 2, τ::Int = 1, base = MathConstants.e)

Construct the embedding

```math
u = \\{{\\bf u}_n \\}_{n = 1}^{N - k + 1} =
\\{[x(i), x(i + 1), \\ldots, x(i + k - 1)]\\}_{n = 1}^{N - k + 1}
```

and use a tree-and-nearest-neighbor search approach to compute

```math
\\phi^k(r) = \\dfrac{1}{N - k + 1} \\sum_{i}^{N - k + 1} \\log_{b}{(C_i^k(r))},
```

taking logarithms to `base` ``b``, and where

```math
C_i^k(r) = \\textrm{number of } j \\textrm{ such that } d({\\bf u}_i, {\\bf u}_j) < r,
```

where ``d`` is the maximum (Chebyshev) distance, `r` is the tolerance, and `N` is the
length of the original scalar-valued time series `x`.
"""
function compute_ϕ(x::AbstractVector{T}, r; m::Int = 2, τ::Int = 1,
        base = MathConstants.e) where T <: Real

    τs = 0:-τ:-(m - 1)*τ
    pts = genembed(x, τs)
    f = length(pts)

    # For each xᵢ ∈ pts, find the indices of neighbors within radius `r` according to
    # the Chebyshev (maximum) metric. Self-inclusion is wanted, so no Theiler window
    # is specified. This tree-approach is more than an order of magnitude
    # faster than checking point-by-point using nested loops, and also allocates
    # more than an order of magnitude less.
    tree = KDTree(pts, Chebyshev())
    idxs_of_neighbors = bulkisearch(tree, pts, WithinRange(r))

    ϕ = 0.0
    for nn_idxsᵢ in idxs_of_neighbors
        # Self-inclusion during the neighbor search means that there are always neighbors,
        # so we never encounter log(0) here.
        Cᵢᵐ = length(nn_idxsᵢ)
        ϕ += log(base, Cᵢᵐ / f)
    end
    return ϕ / f
end
