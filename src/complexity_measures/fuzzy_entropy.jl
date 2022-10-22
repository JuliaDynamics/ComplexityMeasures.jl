using Statistics
using Distances
using Neighborhood
using DelayEmbeddings

export FuzzyEntropy

"""
    FuzzyEntropy([x]; r = 0.2std(x), kwargs...)

An estimator for the fuzzy entropy (Chen et al., 2007)[^Chen2007] complexity/irregularity
measure, used with [`complexity`](@ref).

The keyword argument `r` is mandatory if an input timeseries `x` is not provided.

## Keyword arguments

- `r::Real`: The radius used when querying for nearest neighbors around points. Its value
    should be determined from the input data, for example as some proportion of the
    standard deviation of the data.
- `m::Int = 1`: The embedding dimension.
- `n::Real = 2`: The "fuzzy power". For `d ∈ [-1, 1]`, the exponential function described
    below peaks at `d == 0`. For fixed `r`, larger `n` broadens the peak and smaller `n`
    narrows the peak.
- `τ::Int = `: The embedding lag.
- `metric`: The metric used to compute distances.

## Description

Fuzzy entropy for an input timeseries `x` is computed as follows:

- Construct two `k`-dimensional embeddings, one for `k = m` and one for `k = m + 1`, as
    follows:

```math
{\\bf x}_i^k = (x(i), x(i+τ), x(i+2τ), \\ldots, x(i+(k-1)\\tau)).
```

- Zero-mean-normalize both embeddings.
- Define the function

```math
\\phi(x, k, r, n) =
\\dfrac{1}{N - (m-1)\\tau}\\sum_{i = 1}^{N-(m-1)\\tau}
\\dfrac{1}{N - (m-1)\\tau - 1}\\sum_{j = 1, j \\neq i}^{N-(m-1)\\tau}  D_{ij}^m(n, r).
```

where ``D_{ij}^m(n, r)`` is the "similarity degree" between vectors
``\\bf x_i ∈ \\mathcal{R^k}`` and ``\\bf y_i ∈ \\mathcal{R^k}``, given radius `r`,
the "fuzzy power" `n`, and ``D_{ij}^m``, which is the distance between ``\\bf x_i`` and
``\\bf y_i`` according to the given `metric` (Chen et al. (2007) uses the Chebyshev metric).
The similarity degree is defined as

```math
D_{ij}^m(n, r) = exp(-(d_{ij}^m)^n/r)
```

- The fuzzy entropy is finally estimated as:

```math
FuzzyEn(x, m, r, n) = \\log(\\phi(x, m, r, n)) - \\log(\\phi(x, m + 1, r, n))
```

!!! note "Flexible embedding lag"
    In the original paper, they fix `τ = 1`. In our implementation, equations are modified
    to account for embeddings with `τ != 1`.

[^Chen2007]: Chen, W., Wang, Z., Xie, H., & Yu, W. (2007). Characterization of surface EMG
    signal based on fuzzy entropy. IEEE Transactions on neural systems and rehabilitation
    engineering, 15(2), 266-272.
"""
Base.@kwdef struct FuzzyEntropy{I, N, M, R} <: ComplexityMeasure
    τ::I = 2
    m::I = 1
    n::N = 2
    metric::M = Chebyshev()
    r::R

    function FuzzyEntropy(τ::I, m::I, n::N, metric::M, r::R) where {I, N, M, R}
        r > 0 || throw(ArgumentError("Need r > 0. Got r=$r."))
        m > 0 || throw(ArgumentError("Need m > 0. Got m=$m."))
        new{I, N, M, R}(τ, m, n, metric, r)
    end
end

function FuzzyEntropy(x::AbstractVector{T}; m = 2, τ = 1, n = 2,
        metric = Chebyshev()) where T
    r = 0.2 * Statistics.std(x)
    FuzzyEntropy(τ, m, n, metric, r)
end

"""
    fuzzy_ϕ(Xᵏ::Dataset{k, T}, n, r) where {k, T} → ϕ::Real

The fuzzy ϕ function, analogous to the ϕ function for the approximate entropy
measure. Input data `Xᵏ` are pre-embedded `k`-dimensional vectors.

We use this function to compare similarity degrees between embeddings differing
by 1 in dimensionality.
"""
function fuzzy_ϕ(Xᵏ::Dataset{k, T}, c::FuzzyEntropy) where {k, T}
    (; τ, m, n, metric, r) = c
    L = length(Xᵏ)
    # TODO: for few points, using `Distances.pairwise` is really quick, but the
    # applications of the Fuzzy is typically on time series with hundreds of thousands
    # of points. Then accessing the `LxL` distance matrix becomes a slowing factor.
    # How much slower? Not sure, need to test. In the meanwhile, we just naively
    # compute distances.
    ϕ = 0.0
    for i in 1:L
        xᵢ = Xᵏ[i]
        for j in 1:L
            if (i != j)
                dᵢⱼ = evaluate(metric, xᵢ, Xᵏ[j])
                ϕ += μ(dᵢⱼ, n, r)
            end
        end
        ϕ /= L - 1 # subtract 1 due to self-exclusion
    end
    ϕ /= L

    return ϕ
end

μ(dᵢⱼᵐ, n, r) = exp(-(dᵢⱼᵐ^n) / r)


function complexity(c::FuzzyEntropy, x::AbstractVector{T}) where T <: Real
    (; τ, m, n, metric, r) = c
    # Standardize the embeddings. Note: in original paper only mean subtraction is done,
    # here we also scale to unit variance.
    Eₘ = standardize(genembed(x, 0:τ:((m - 1) * τ)))
    Eₘ₊₁ = standardize(genembed(x, 0:τ:(m * τ)))
    fϕₘ = fuzzy_ϕ(Eₘ, c)
    fϕₘ₊₁ = fuzzy_ϕ(Eₘ₊₁, c)

    return log(fϕₘ) - log(fϕₘ₊₁)
end
