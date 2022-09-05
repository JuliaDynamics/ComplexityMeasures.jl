using DelayEmbeddings
using NearestNeighbors
using StatsBase

export sample_entropy
export SampleEntropy

# See comment in https://github.com/JuliaDynamics/Entropies.jl/pull/71 for why
# inrangecount is used and not NeighborHood.bulkisearch.
"""
    computeprobs(x; k::Int, m::Int, r, metric = Chebyshev())

Compute the probabilities required for [`sample_entropy`](@ref). `k` is the embedding
dimension. `m` is a normalization constant. `r` is the radius.
"""
function computeprobs(x; k::Int, m::Int, r, metric = Chebyshev())
    N = length(x)
    pts = genembed(x, 0:(k - 1))

    # For each `k`-dimensional xᵢ ∈ pts, locate its within-range-`r` nearest neighbors,
    # excluding the point `xₖ` as a neighbor to itself.
    tree = KDTree(pts, metric)

    # inrangecount includes the point itself, so subtract 1
    cts = [inrangecount(tree, pᵢ, r) - 1 for pᵢ in pts]

    # Pᵐ := The probability that two sequences will match for k points
    Pᵐ = 0
    c = N - m - 1
    for ct in cts
        Pᵐ += ct / c
    end
    Pᵐ /= N - m

    return Pᵐ
end

"""
    SampleEntropy(; r = 0.1, m::Int = 2, normalize = false)

Estimate the sample entropy (Richman & Moorman, 2000)[^Richman2000]

```math
SampEn(m, r) = \\lim_{N \\to \\infty} \\left[ -\\ln \\dfrac{A^{m+1}(r)}{B^m(r)} \\right].
```

If `normalize == true`, then the sample entropy is normalized to `[0, 1]`,
based on the possible range of values it can take (details in Richman & Moorman, 2000).
Always uses the natural logarithm, so `base` is ignored.

## Estimation

Assume the input data is a time series `y`.
To estimate ``SampEn(m,r)``, first construct from `y` the `N-m-1` possible `m`-dimensional
embedding vectors ``{\\bf x}_i^m = (x(i), x(i+1), \\ldots, x(i+m-1))``. Next, compute
``B^{m}(r) = \\sum_{i = 1}^{N-m} B_i^{m}(r)``, where ``B_i^{m}(r)`` is the
number of vectors within radius `r` of ``{\\bf x}_i`` (without self-inclusion).

Finally, repeat the procedure, but with embedding vectors of dimension `m + 1`, and
compute ``A^{m+1}(r) = \\sum_{i = 1}^{N-m} A_i^{m+1}(r)``, where ``A_i^{m+1}(r)`` is the
number of vectors within radius `r` of ``{\\bf x}_i^{m+1}``.

Sample entropy is then estimated as

```math
SampEn(m,r) = -\\ln{\\dfrac{A^{m+1}(r)}{B^{m}(r)}}.
```

## Data requirements

If the radius `r` is too small relative to the magnitudes of the `xᵢ ∈ x`, or if `x` is
too short, it is possible that no radius-`r` neighbors are found, so that
`SampEn(m,r) = log(0)`. If logarithms of zeros are encountered, `0.0` is
returned.

!!! info
    This estimator is only available for entropy estimation.
    Probabilities cannot be obtained directly.

[^Richman2000]: Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using approximate entropy and sample entropy. American Journal of Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
"""
Base.@kwdef struct SampleEntropy <: EntropyEstimator
    r::Real = 0.1
    m::Int = 2
    metric = Chebyshev()
    normalize::Bool = false
end

function scale(x, min_range, max_range, min_target, max_target)
    (x - min_range)/(max_range - min_range) * (max_target - min_target) + min_target
end

function genentropy(x::AbstractVector{T}, est::SampleEntropy; base = nothing) where T <: Real
    m, r, metric, normalize = est.m, est.r, est.metric, est.normalize
    Aᵐ⁺¹ = computeprobs(x; k = m + 1, m = m, r = r, metric = metric)
    Bᵐ = computeprobs(x; k = m, m = m, r = r, metric = metric)

    # We may want to handle these special cases where the sample entropy isn't defined.
    # Not sure what is the best approach, so we just return zero for now.
    sampen = 0.0

    # Conditional probability of 0 -> SampEn = ∞
    if Aᵐ⁺¹ == 0.0
        sampen = 0.0
    # No regularity has been detected
    elseif Bᵐ == 0.0
        sampen = 0.0
    else
        sampen = -log(Aᵐ⁺¹ / Bᵐ)
    end

    if (normalize)
        # Richman & Moorman provide these constraints
        # for the possible nonzero values of the sample entropy.
        N = length(x)
        lowerbound = 2*((N - m - 1)*(N - m))^(-1)
        upperbound = log(N - m) + log(N - m - 1) - log(2)

        # Scale to unit interval.
        sampen = scale(sampen, lowerbound, upperbound, 0.0, 1.0)
    end

    return sampen
end

function genentropy(x::AbstractDataset, est::SampleEntropy; base = nothing)
    throw(
        ArgumentError("Sample entropy is currently not defined for multivariate data.")
    )
end

"""
    sample_entropy(x; m = 2, r = StatsBase.std(x), normalize = false) → SampEn::Float64

Shorthand for `genentropy(x, SampleEntropy(m = m, r = r, normalize = normalize))` which
calculates the sample entropy for dimension `m` with tolerance radius `r`, normalizing
the estimate to `[0, 1]` if `normalize == true`.

## Examples

```jldoctest; setup = :(using Entropies)
julia> x = repeat([0.84, 0.52, 0.46], 1000);
julia> hx = sample_entropy(x, m = 2, r = 0.3)
(0.21833796796344462, 0.3727646139487059)
```
"""
function sample_entropy(x; m = 2, r = 0.2 * StatsBase.std(x), metric = Chebyshev(),
        normalize = false)

    est = SampleEntropy(m = m, r = r, normalize = normalize, metric = metric)
    return genentropy(x, est)
end
