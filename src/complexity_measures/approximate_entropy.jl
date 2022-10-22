using DelayEmbeddings
using Neighborhood: inrangecount
using Distances
using Statistics

export ApproxEntropy
export approx_entropy

"""
    ApproxEntropy([x]; r = 0.2std(x), kwargs...)

An estimator for the approximate entropy (ApEn; Pincus, 1991)[^Pincus1991] complexity
measure, used with [`complexity`](@ref).

The keyword argument `r` is mandatory if an input timeseries `x` is not provided.

## Keyword arguments
- `r::Real`: The radius used when querying for nearest neighbors around points. Its value
    should be determined from the input data, for example as some proportion of the
    standard deviation of the data.
- `m::Int = 2`: The embedding dimension.
- `τ::Int = 1`: The embedding lag.
- `base::Real = MathConstants.e`: The base to use for the logarithm. Pincus (1991) uses the
    natural logarithm.
- `metric`: The metric used to compute distances.

## Description

Approximate entropy is defined as

```math
ApEn(m ,r) = \\lim_{N \\to \\infty} \\left[ \\phi(x, m, r) - \\phi(x, m + 1, r) \\right].
```

Approximate entropy is estimated for a timeseries `x`, by first embedding `x` using
embedding dimension `m` and embedding lag `τ`, then searching for similar vectors within
tolerance radius `r`, using the estimator described below, with logarithms to the given
`base` (natural logarithm is used in Pincus, 1991).

Specifically, for a finite-length timeseries `x`, an estimator for ``ApEn(m ,r)`` is

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

!!! note "Flexible embedding lag"
    In the original paper, they fix `τ = 1`. In our implementation, the normalization
    constant is modified to account for embeddings with `τ != 1`.

[^Pincus1991]: Pincus, S. M. (1991). Approximate entropy as a measure of system complexity.
    Proceedings of the National Academy of Sciences, 88(6), 2297-2301.
"""
Base.@kwdef struct ApproxEntropy{I, M, B, R} <: ComplexityMeasure
    m::I = 2
    τ::I = 1
    metric::M = Chebyshev()
    base::B = MathConstants.e
    r::R

    function ApproxEntropy(m::I, τ::I, r::R, base::B, metric::M) where {I, R, M, B}
        m >= 1 || throw(ArgumentError("m must be >= 1. Got m=$(m)."))
        r > 0 || throw(ArgumentError("r must be > 0. Got r=$(r)."))
        new{I, M, B, R}(m, τ, metric, base, r)
    end
end

function ApproxEntropy(x::AbstractVector{T}; m::Int = 2, τ::Int = 1, metric = Chebyshev(),
        base = MathConstants.e) where T
    r = 0.2 * Statistics.std(x)
    ApproxEntropy(m, τ, r, base, metric)
end

function ApproxEntropy(; r, m::Int = 2, τ::Int = 1, metric = Chebyshev())
    ApproxEntropy(m, τ, r, base, metric)
end

function complexity(c::ApproxEntropy, x::AbstractVector{T}) where T
    (; m, τ, r, base) = c

    # Definition in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7515030/
    if m == 1
       return compute_ϕ(x; k = m, r, τ, base)
    else
        ϕᵐ = compute_ϕ(x; k = m, r, τ, base)
        ϕᵐ⁺¹ = compute_ϕ(x; k = m + 1, r, τ, base)
        return ϕᵐ - ϕᵐ⁺¹
    end
end

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
    tree = KDTree(pts, Chebyshev())

    # Account for `τ != 1` in the normalization constant.
    f = length(x) - (k - 1)*τ

    # `inrangecount` counts the query point itself, which is wanted for approximate entropy,
    # because there are always neighbors and thus log(0) is never encountered.
    ϕ = sum(log(base, inrangecount(tree, pᵢ, r) / f) for pᵢ in pts)

    return ϕ / f
end

"""
    approx_entropy(x; m = 2, τ = 1, r = 0.2 * Statistics.std(x), base = MathConstants.e)

Convenience syntax for computing the approximate entropy (Pincus, 1991) for timeseries `x`.

This is just a wrapper for `complexity(ApproxEntropy(; m, τ, r, base), x)` (see
also [`ApproxEntropy`](@ref)).
"""
function approx_entropy(x; m = 2, τ = 1, r = 0.2 * Statistics.std(x),
         base = MathConstants.e)
    c = ApproxEntropy(; m, τ, r, base)
    return complexity(c, x)
end
