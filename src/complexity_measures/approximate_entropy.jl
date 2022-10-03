using DelayEmbeddings
using Neighborhood
using Distances
using Statistics

export approx_entropy

"""
    approx_entropy(x; m = 2, τ = 1, r = 0.2 * Statistics.std(x), base = MathConstants.e)

Compute the approximate entropy (ApEn; Pincus, 1991)[^Pincus1991] of a univariate
timeseries `x`, by first embedding `x` using embedding dimension `m` and embedding lag `τ`,
then searching for similar vectors within tolerance radius `r`, using the estimator
described below, with logarithms to the given `base` (natural logarithm is used in
Pincus, 1991).

`r` should be determined from the input data, for example as some fraction of the
standard deviation of the input.

## Description

Approximate entropy is defined as

```math
ApEn(m ,r) = \\lim_{N \\to \\infty} \\left[ \\phi(x, m, r) - \\phi(x, m + 1, r) \\right].
```

For a finite-length timeseries `x`, an estimator for ``ApEn(m ,r)`` is

```math
ApEn(m, r, N) = \\phi(x, m, r, N) -  \\phi(x, m + 1, r, N),
```

where `N = length(x)` and

```math
\\phi(x, k, r, N) =
\\dfrac{1}{N-(k-1)\\tau} \\sum_{i=1}^{N - (k-1)\\tau}
\\log{\\left(
    \\sum_{j = 1}^{N-(k-1)\\tau} \\dfrac{\\theta(d({\\bf x}_i^m, {\\bf x}_j^m) \\leq r)}{N-(k-1)\\tau}
    \\right)}.
```

Here, ``\\theta(\\cdot)`` returns 1 if the argument is true and 0 otherwise,
 ``d({\\bf x}_i, {\\bf x}_j)`` returns the Chebyshev distance between vectors
 ``{\\bf x}_i`` and ``{\\bf x}_j``, and the `k`-dimensional embedding vectors are
constructed from the input timeseries ``x(t)`` as

```math
{\\bf x}_i^k = (x(i), x(i+τ), x(i+2τ), \\ldots, x(i+(k-1)\\tau)).
```

Note: in the original paper, they fix `τ = 1`; in our implementation, the normalization
constant is modified to account for embeddings with `τ != 1`.

[^Pincus1991]: Pincus, S. M. (1991). Approximate entropy as a measure of system complexity.
    Proceedings of the National Academy of Sciences, 88(6), 2297-2301.
"""
function approx_entropy(x::AbstractVector{T}; m::Int = 2, τ::Int = 1,
        r = 0.2 * std(x), base = MathConstants.e) where T
    m >= 1 || throw(ArgumentError("m must be >= 1. Got m=$(m)."))

    # Definition in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7515030/
    if m == 1
       return compute_ϕ(x; k = m, r, τ, base)
    else
        ϕᵐ = compute_ϕ(x; k = m, r, τ, base)
        ϕᵐ⁺¹ = compute_ϕ(x; k = m + 1, r, τ, base)

        return ϕᵐ - ϕᵐ⁺¹
    end
end

approx_entropy(x; kwargs...) =
    throw(ArgumentError(
        "`approx_entropy` not implemented for input data of type $(typeof(x))"
    ))

"""
    compute_ϕ(x::AbstractVector{T}; r = 0.2 * Statistics.std(x), k::Int = 2,
        τ::Int = 1, base = MathConstants.e) where T <: Real

Construct the embedding

```math
u = \\{{\\bf u}_n \\}_{n = 1}^{N - k + 1} =
\\{[x(i), x(i + 1), \\ldots, x(i + k - 1)]\\}_{n = 1}^{N - k + 1}
```

and use a tree-and-nearest-neighbor search approach to compute

```math
\\phi^k(r) = \\dfrac{1}{N - kτ + 1} \\sum_{i}^{N - kτ + 1} \\log_{b}{(C_i^k(r))},
```

taking logarithms to `base` ``b``, and where

```math
C_i^k(r) = \\textrm{number of } j \\textrm{ such that } d({\\bf u}_i, {\\bf u}_j) < r,
```

where ``d`` is the maximum (Chebyshev) distance, `r` is the tolerance, and `N` is the
length of the original scalar-valued time series `x`.
"""
function compute_ϕ(x::AbstractVector{T}; r = 0.2 * Statistics.std(x), k::Int = 2,
        τ::Int = 1, base = MathConstants.e) where T <: Real
    τs = 0:τ:(k - 1)*τ
    pts = genembed(x, τs)

    N = length(x)
    # The original normalization uses `τ = 1`. This needs to be adjusted to account for
    # `τ != 1` in the embedding.
    f = N - (k-1)*τ

    # For each xᵢ ∈ pts, find the indices of neighbors within radius `r` according to
    # the Chebyshev (maximum) metric. Self-inclusion is wanted, so no Theiler window
    # is specified. This tree-approach is more than an order of magnitude
    # faster than checking point-by-point using nested loops, and also allocates
    # more than an order of magnitude less.
    tree = KDTree(pts, Chebyshev())
    idxs_of_neighbors = bulkisearch(tree, pts, WithinRange(r))

    # Self-inclusion during the neighbor search means that there are always neighbors,
    # so we never encounter log(0) here.
    ϕ = sum(log(base, length(neighborsᵢ) / f) for neighborsᵢ in idxs_of_neighbors)

    return ϕ / f
end
