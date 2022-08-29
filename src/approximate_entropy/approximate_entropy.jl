using DelayEmbeddings
using Neighborhood
using StatsBase
using Distances

export approx_entropy

"""
    approx_entropy(x, m = 3, r = 0.5 * StatsBase.std(x), base = MathConstants.e) → ApEn

Compute the approximate entropy (ApEn; Pincus, 1991)[^Pincus1991] of the scalar-valued
time series `x`, using embedding dimension (pattern length) `m` and tolerance radius `r`,
using logarithms to the given `base`.

[^Pincus1991]: Pincus, S. M. (1991). Approximate entropy as a measure of system complexity. Proceedings of the National Academy of Sciences, 88(6), 2297-2301.
"""
function approx_entropy(x; m = 2, r = 0.2 * StatsBase.std(x), base = MathConstants.e)
    m >= 1 || throw(ArgumentError("m must be >= 1. Got m=$m."))
    N = length(x)
    ϕᵐ = ϕmr_tree(x, m, r, N, base = base)
    ϕᵐ⁺¹ = ϕmr_tree(x, m + 1, r, N, base = base)
    return ϕᵐ - ϕᵐ⁺¹
end

"""
    ϕmr_tree(x::AbstractVector{T}, k, r, N; base = MathConstants.e)

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

## Examples

```jldoctest; setup = :(using Entropies)
julia> x = repeat([2, 3, 4, 8, 8, 9, 3, 6, 4, 5, 8, 7], 100);

julia> approx_entropy(x, m = 2, r = 3)
0.42020216510849995
```
"""
function ϕmr_tree(x::AbstractVector{T}, k, r, N::Int;
        base = MathConstants.e) where T <: Real
    pts = genembed(x, 0:(k - 1))

    # For each xᵢ ∈ pts, find the indices of neighbors within radius `r` according to
    # the Chebyshev (maximum) metric. Self-inclusion is wanted, so no Theiler window
    # is specified. This tree-approach is more than an order of magnitude
    # faster than checking point-by-point using nested loops, and also allocates
    # more than an order of magnitude less.
    tree = KDTree(pts, Chebyshev())
    idxs_of_neighbors = bulkisearch(tree, pts, WithinRange(r))

    q = N - k + 1
    ϕ = 0.0
    for nn_idxsᵢ in idxs_of_neighbors
        # Self-inclusion during the neighbor search means that there are always neighbors,
        # so we never encounter log(0) here.
        ϕ += log(base, length(nn_idxsᵢ) / q)
    end
    return ϕ / q
end
