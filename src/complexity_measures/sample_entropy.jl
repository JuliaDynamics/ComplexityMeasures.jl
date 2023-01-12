using DelayEmbeddings
using Statistics
using Neighborhood.NearestNeighbors: Chebyshev, KDTree
using Neighborhood: inrangecount

export SampleEntropy
export entropy_sample

"""
    SampleEntropy([x]; r = 0.2std(x), kwargs...) <: ComplexityEstimator

An estimator for the sample entropy complexity measure (Richman & Moorman,
2000)[^Richman2000], used with [`complexity`](@ref) and [`complexity_normalized`](@ref).

The keyword argument `r` is mandatory if an input timeseries `x` is not provided.

## Keyword arguments

- `r::Real`: The radius used when querying for nearest neighbors around points. Its value
    should be determined from the input data, for example as some proportion of the
    standard deviation of the data.
- `m::Int = 1`: The embedding dimension.
- `τ::Int = 1`: The embedding lag.

## Description

An *estimator* for sample entropy using radius `r`, embedding dimension `m`, and
embedding lag `τ` is

```math
SampEn(m,r, N) = -\\ln{\\dfrac{A(r, N)}{B(r, N)}}.
```

Here,

```math
\\begin{aligned}
B(r, m, N) = \\sum_{i = 1}^{N-m\\tau} \\sum_{j = 1, j \\neq i}^{N-m\\tau} \\theta(d({\\bf x}_i^m, {\\bf x}_j^m) \\leq r) \\\\
A(r, m, N) = \\sum_{i = 1}^{N-m\\tau} \\sum_{j = 1, j \\neq i}^{N-m\\tau} \\theta(d({\\bf x}_i^{m+1}, {\\bf x}_j^{m+1}) \\leq r) \\\\
\\end{aligned},
```

where ``\\theta(\\cdot)`` returns 1 if the argument is true and 0 otherwise,
and ``d(x, y)`` computes the Chebyshev distance between ``x`` and ``y``,
and  ``{\\bf x}_i^{m}`` and ``{\\bf x}_i^{m+1}`` are `m`-dimensional and
`m+1`-dimensional embedding vectors, where `k`-dimensional embedding vectors are constructed
from the input timeseries ``x(t)`` as

```math
{\\bf x}_i^k = (x(i), x(i+τ), x(i+2τ), \\ldots, x(i+(k-1)\\tau)).
```

Quoting Richman & Moorman (2002): "SampEn(m,r,N) will be defined except when B = 0,
in which case no regularity has been detected, or when A = 0, which corresponds to
a conditional probability of 0 and an infinite value of SampEn(m,r,N)".
In these cases, `NaN` is returned.

If computing the normalized measure, then the resulting sample entropy is on `[0, 1]`.

!!! note "Flexible embedding lag"
    The original algorithm fixes `τ = 1`. All formulas here are modified to account for
    any `τ`.

See also: [`entropy_sample`](@ref).

[^Richman2000]: Richman, J. S., & Moorman, J. R. (2000). Physiological time-series
    analysis using approximate entropy and sample entropy. American Journal of
    Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
"""
Base.@kwdef struct SampleEntropy{R} <: ComplexityEstimator
    m::Int = 2
    τ::Int = 1
    r::R

    function SampleEntropy(m::Int, τ::Int, r::R) where {R}
        m >= 1 || throw(ArgumentError("m must be >= 1. Got m=$(m)."))
        r > 0 || throw(ArgumentError("r must be > 0. Got r=$(r)."))
        new{R}(m, τ, r)
    end
    function SampleEntropy(x::AbstractVector; m::Int = 2, τ::Int = 1)
        r = 0.2 * Statistics.std(x)
        SampleEntropy(m, τ, r)
    end
end

# See comment in https://github.com/JuliaDynamics/ComplexityMeasures.jl/pull/71 for why
# inrangecount is used and not NeighborHood.bulkisearch.
"""
    sample_entropy_probs(x; k::Int = 2, m::Int = 2, τ::Int = 1, r = 0.2 * Statistics.std(x))

Compute the probabilities required for [`entropy_sample`](@ref). `k` is the embedding
dimension, `τ` is the embedding lag, and `m` is a normalization constant (so that we
consider the same number of points for both the `m`-dimensional and the `m+1`-dimensional
embeddings), and `r` is the radius.
"""
function sample_entropy_probs(x; k::Int = 2, m::Int = 2, τ::Int = 1, r = 0.2 * Statistics.std(x))

    N = length(x)
    pts = genembed(x, 0:τ:(k - 1)*τ)
    tree = KDTree(pts, Chebyshev())

    # Pᵐ := The probability that two sequences will match for k points.
    # We only consider the first N-m*τ vectors, regardless of embedding dimension. This
    # means that the last vector is skipped in the highest-dimensional embedding.
    # `inrangecount` includes the point itself, so subtract 1.
    Pᵐ = sum(inrangecount(tree, pᵢ, r) - 1 for pᵢ in Iterators.take(pts, N - m*τ))

    # We don't include the normalization terms here, because they cancel in the final
    # computation.
    return Pᵐ
end

function scale(x, min_range, max_range, min_target, max_target)
    (x - min_range)/(max_range - min_range) * (max_target - min_target) + min_target
end

function complexity(c::SampleEntropy, x::AbstractVector{T}) where T <: Real
    (; m, τ, r) = c

    A = sample_entropy_probs(x; m = m, τ = τ, r = r, k = m + 1)
    B = sample_entropy_probs(x; m = m, τ = τ, r = r, k = m)

    if A == 0.0 || B == 0.0
        return NaN
    else
        return -log(A / B)
    end
end

function complexity_normalized(c::SampleEntropy, x::AbstractVector{T}) where T <: Real
    (; m, τ, r) = c

    sampen = complexity(c, x)
    if isnan(sampen) || isinf(sampen)
        return sampen
    else
        # Richman & Moorman (2000) provide constraints for the possible nonzero values
        # of the sample entropy. We use these values to scale sample entropy to the
        # unit interval.
        # Here, the constraints have been modified to account for `τ != 1`.
        # The `N - m*τ` terms account for the outer sums, while the `N - m*τ - 1`
        # terms account for the inner sums (subtracting one due to self-exclusion).
        # For τ = 1, this recovers the normalization from Richman & Moorman (2000).
        # The absolute value accounts for negative lags.
        N = length(x)
        lowerbound = 1/(2*(N - m*abs(τ) - 1) * (N - m*abs(τ)))
        upperbound = log(N - m*abs(τ)) + log(N - m*abs(τ) - 1) - log(2)
        return scale(sampen, lowerbound, upperbound, 0.0, 1.0)
    end
end
