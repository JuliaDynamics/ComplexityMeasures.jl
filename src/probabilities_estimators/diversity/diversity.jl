using DelayEmbeddings

export Diversity

function cosine_similarity(xᵢ, xⱼ)
    return sum(xᵢ .* xⱼ) / (sqrt(sum(xᵢ .^ 2)) * sqrt(sum(xⱼ .^ 2)))
end

"""
    Diversity(; m::Int, τ::Int, nbins::Int)

A [`ProbabilitiesEstimator`](@ref) based on the cosine similarity. It can be used with [`entropy`](@ref) to
compute diversity entropy (Wang et al., 2020)[^Wang2020].

The implementation here allows for `τ != 1`, which was not considered in the original paper.

## Description

Diversity probabilities are computed as follows.

1. From the input time series `x`, using embedding lag `τ` and embedding dimension `m`,
    construct the embedding
    ``Y = \\{\\bf x_i \\} = \\{(x_{i}, x_{i+\\tau}, x_{i+2\\tau}, \\ldots, x_{i+m\\tau - 1}\\}_{i = 1}^{N-mτ}``.
2. Compute ``D = \\{d(\\bf x_t, \\bf x_{t+1}) \\}_{t=1}^{N-mτ-1}``,
    where ``d(\\cdot, \\cdot)`` is the cosine similarity between two `m`-dimensional
    vectors in the embedding.
3. Divide the interval `[-1, 1]` into `nbins` equally sized subintervals.
4. Construct a histogram of cosine similarities ``d \\in D`` over those subintervals.
5. Sum-normalize the histogram to obtain probabilities.

## Implements

- [`probabilities_and_outcomes`](@ref). Events are the corners of the cosine similarity bins.
    Each bin has width `nextfloat(2 / nbins)`.
- [`total_outcomes`](@ref). The total number of states is given by `nbins`.

[^Wang2020]: 
    Wang, X., Si, S., & Li, Y. (2020). Multiscale diversity entropy: A novel
    dynamical measure for fault diagnosis of rotating machinery. IEEE Transactions on
    Industrial Informatics, 17(8), 5419-5429.
"""
Base.@kwdef struct Diversity <: ProbabilitiesEstimator
    m::Int = 2
    τ::Int = 1 # Note: the original paper does not allow τ != 1
    nbins::Int = 5
end

function similarities_and_binning(x::AbstractVector{T}, est::Diversity) where T <: Real
    τs = 0:est.τ:(est.m - 1)*est.τ
    Y = genembed(x, τs)

    ds = zeros(Float64, length(Y) - 1)
    for (i, (yᵢ, yᵢ₊₁)) in enumerate(zip(Y.data[1:end-1], Y.data[2:end]))
        ds[i] = cosine_similarity(yᵢ, yᵢ₊₁)
    end

    # Cosine similarities are all on [-1.0, 1.0], so discretize this interval.
    binning = FixedRectangularBinning(-1.0, 1.0, est.nbins)

    return ds, binning
end

function probabilities(x::AbstractVector{T}, est::Diversity) where T <: Real
    ds, binning = similarities_and_binning(x, est)
    return probabilities(ds, binning)
end

function probabilities_and_outcomes(x::AbstractVector{T}, est::Diversity) where T <: Real
    ds, binning = similarities_and_binning(x, est)
    return probabilities_and_outcomes(ds, binning)
end

total_outcomes(est::Diversity) = est.nbins
