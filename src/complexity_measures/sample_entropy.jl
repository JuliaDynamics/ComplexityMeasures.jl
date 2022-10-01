using DelayEmbeddings
using Statistics
using Neighborhood.NearestNeighbors: inrangecount, Chebyshev

export sample_entropy

# See comment in https://github.com/JuliaDynamics/Entropies.jl/pull/71 for why
# inrangecount is used and not NeighborHood.bulkisearch.
"""
    computeprobs(x; k::Int = 2, m::Int = 2, τ::Int = 1, r = 0.2 * Statistics.std(x),
        metric = Chebyshev()

Compute the probabilities required for [`sample_entropy`](@ref). `k` is the embedding
dimension, `τ` is the embedding lag, and `m` is a normalization constant (so that we
consider the same number of points for both the `m`-dimensional and the `m+1`-dimensional
embeddings), and `r` is the radius.
"""
function computeprobs(x; k::Int = 2, m::Int = 2, τ::Int = 1, r = 0.2 * Statistics.std(x),
        metric = Chebyshev())

    N = length(x)
    pts = genembed(x, 0:τ:(k - 1)*τ)
    # For each `k`-dimensional xᵢ ∈ pts, locate its within-range-`r` nearest neighbors,
    # excluding the point `xₖ` as a neighbor to itself.
    tree = KDTree(pts, metric)

    # Pᵐ := The probability that two sequences will match for k points.
    # We only consider the first N-m*τ vectors, regardless of embedding dimension. This
    # means that the last vector is skipped in the highest-dimensional embedding.
    # `inrangecount`` includes the point itself, so subtract 1.
    Pᵐ = sum(inrangecount(tree, pᵢ, r) - 1 for pᵢ in Iterators.take(pts, N - m*τ))

    # We don't include the normalization terms here, because they cancel in the final
    # computation.
    return Pᵐ
end


function scale(x, min_range, max_range, min_target, max_target)
    (x - min_range)/(max_range - min_range) * (max_target - min_target) + min_target
end

"""
    sample_entropy(x; r::Real = 0.2 * Statistics.std(x), m::Int = 2, τ::Int = 1,
        metric = Chebyshev(), normalize::Bool = true) → SampEn::Real

Estimate the sample entropy (Richman & Moorman, 2000)[^Richman2000] for the univariate
timeseries `x`, using radius `r`, embedding dimension `m`, embedding lag `τ` (note: the
original algorithm fixes `τ = 1`; all formulas here are modified to account for any `τ`),

If `normalize == true`, then the sample entropy is normalized to `[0, 1]`,
based on the possible range of values it can take (details in Richman & Moorman, 2000).

## Description

Sample entropy is a complexity measure defined as

```math
SampEn(m, r) = \\lim_{N \\to \\infty} \\left[ -\\ln \\dfrac{A^{m+1}(r)}{B^m(r)} \\right],
```

An *estimator* for sample entropy is

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
and ``d(x, y)`` computes the distance between ``x`` and ``y`` according to `metric`
(default is Chebyshev), and  ``{\\bf x}_i^{m}`` and ``{\\bf x}_i^{m+1}`` are `m`-dimensional and
`m+1`-dimensional embedding vectors, where `k`-dimensional embedding vectors are constructed
from the input timeseries ``x(t)`` as

```math
{\\bf x}_i^k = (x(i), x(i+τ), x(i+2τ), \\ldots, x(i+(k-1)\\tau)).
```

Quoting Richman & Moorman (2002): "SampEn(m,r,N) will be defined except when B = 0,
in which case no regularity has been detected, or when A = 0, which corresponds to
a conditional probability of 0 and an infinite value of SampEn(m,r,N)".
In these cases, `NaN` is returned.

[^Richman2000]: Richman, J. S., & Moorman, J. R. (2000). Physiological time-series
    analysis using approximate entropy and sample entropy. American Journal of
    Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
"""
function sample_entropy(x::AbstractVector{T}; r::Real = 0.2 * Statistics.std(x),
        m::Int = 2, τ::Int = 1,
        metric = Chebyshev(),
        normalize::Bool = false
        ) where T <: Real

    A = computeprobs(x; m = m, τ = τ, r = r, metric = metric, k = m + 1)
    B = computeprobs(x; m = m, τ = τ, r = r, metric = metric, k = m)

    # Conditional probability of 0 -> SampEn = ∞, or no regularity has been detected.
    # In these cases, we skip normalization.
    if A == 0.0 || B == 0.0
        return NaN
    else
        sampen = -log(A / B)
        if normalize
            # Richman & Moorman (2000) provide constraints for the possible nonzero values
            # of the sample entropy. We use these values to scale sample entropy to the
            # unit interval.
            # Here, the constraints have been modified to account for `τ != 1`.
            # The `N - m*τ` terms account for the outer sums, while the `N - m*τ - 1`
            # terms account for the inner sums (subtracting one due to self-exclusion).
            # For τ = 1, this recovers the normalization from Richman & Moorman (2000).
            # The absolute value accounts for negative lags.
            N = length(x)
            lowerbound = 2*((N - m*abs(τ) - 1)*(N - m*abs(τ)))^(-1)
            upperbound = log(N - m*abs(τ)) + log(N - m*abs(τ) - 1) - log(2)
            sampen = scale(sampen, lowerbound, upperbound, 0.0, 1.0)
        end

        return sampen
    end
end

sample_entropy(x; kwargs...) =
    throw(ArgumentError(
        "Sample entropy is currently not defined for input of type $(typeof(x))."
    ))
