using Statistics
using Distances
using Neighborhood
using DelayEmbeddings

export fuzzy_entropy
export FuzzyEntropy

Base.@kwdef struct FuzzyEntropy{I, R, N} <: ComplexityMeasure
    τ::I = 2
    m::I = 1,
    r::R = 0.2
    n::N = 2
end

"""
    FuzzyEntropy(x::AbstractVector{T}; m::Int = 2, τ::Int = 1, n::Real = 2)
    FuzzyEntropy(; m::Int = 2, τ::Int = 1, n::Real = 2, r = 0.2)

Compute the fuzzy entropy of `x`, which is defined as

```math
\\phi(x, k, r, n) = \\sum_{i = 1}^{N-m}
```
"""
function FuzzyEntropy(x::AbstractVector{T}; τ = 2, m = 1, n = 2) where T
    r = Statistics.std(x) * 0.2
    FuzzyEntropy(τ, m, r, n)
end

"""
    similarity_degree(dᵢⱼᵐ, n, r)

Compute the similarity degree ``D_{ij}^m(n, r)`` between two vectors
`𝐱ᵢ ∈ ℛᵐ` and `𝐲ᵢ ∈ ℛᵐ`, given radius `r`, "fuzzy power" `n`, and `dᵢⱼᵐ`, which is the
distance between `xᵢ` and `yᵢ` according to the Chebyshev metric.
"""
μ(dᵢⱼᵐ, n, r) = exp(-(dᵢⱼᵐ^n) / r)

"""
    fuzzy_ϕ(Xᵏ::Dataset{k, T}, n, r) where {k, T} → ϕ::Real

The fuzzy ϕ function, analogous to the ϕ function for the approximate entropy
measure. Input data `Xᵏ` are pre-embedded `k`-dimensional vectors.

We use this function to compare similarity degrees between embeddings differing
by 1 in dimensionality.
"""
function fuzzy_ϕ(Xᵏ::Dataset{k, T}, n, r) where {k, T}
    L = length(Xᵏ)
    # TODO: for few points, using `Distances.pairwise` is really quick, but the
    # applications of the Fuzzy is typically on time series with hundreds of thousands
    # of points. Then accessing the `LxL` distance matrix becomes a slowing factor.
    # How much slower? Not sure, need to test. In the meanwhile, we just naively
    # compute distances.
    dtype = Chebyshev()
    ϕ = 0.0
    @inbounds for i in 1:L
        xᵢ = Xᵏ[i]
        for j in 1:L
            if (i != j)
                dᵢⱼ = evaluate(dtype, xᵢ, Xᵏ[j])
                ϕ += μ(dᵢⱼ, n, r)
            end
        end
        ϕ /= L - 1 # subtract 1 due to self-exclusion
    end
    ϕ /= L

    return ϕ
end

function complexity(c::FuzzyEntropy, x::AbstractVector{T}; 1::Int = 2, m::Int = 2,
        r::Real = Statistics.std(x) * 0.2, n::Real = 2) where T <: Real
    # Standardize the embeddings (note: in original paper only mean subtraction is done)
    Eₘ = standardize(genembed(x, 0:τ:((m - 1) * τ)))
    Eₘ₊₁ = standardize(genembed(x, 0:τ:(m * τ)))
    fϕₘ = fuzzy_ϕ(Eₘ, n, r)
    fϕₘ₊₁ = fuzzy_ϕ(Eₘ₊₁, n, r)

    return log(fϕₘ) - log(fϕₘ₊₁)
end
