using DelayEmbeddings

export Diversity

"""
    Diversity(; m::Int, τ::Int, nbins::Int)

A [`ProbabilitiesEstimator`](@ref) based on the cosine similarity.
It can be used with [`information`](@ref) to
compute the diversity entropy of an input timeseries[^Wang2020].

The implementation here allows for `τ != 1`, which was not considered in the original paper.

## Description

Diversity probabilities are computed as follows.

1. From the input time series `x`, using embedding lag `τ` and embedding dimension `m`,
    construct the embedding
    ``Y = \\{\\bf x_i \\} = \\{(x_{i}, x_{i+\\tau}, x_{i+2\\tau}, \\ldots, x_{i+m\\tau - 1}\\}_{i = 1}^{N-mτ}``.
2. Compute ``D = \\{d(\\bf x_t, \\bf x_{t+1}) \\}_{t=1}^{N-mτ-1}``,
    where ``d(\\cdot, \\cdot)`` is the cosine similarity between two `m`-dimensional
    vectors in the embedding.
3. Divide the interval `[-1, 1]` into `nbins` equally sized subintervals
   (including the value `+1`).
4. Construct a histogram of cosine similarities ``d \\in D`` over those subintervals.
5. Sum-normalize the histogram to obtain probabilities.

## Outcome space
The outcome space for `Diversity` is the bins of the `[-1, 1]` interval,
and the return configuration is the same as in [`ValueHistogram`](@ref) (left bin edge).

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

function probabilities(est::Diversity, x::AbstractVector{T}) where T <: Real
    ds, rbc = similarities_and_binning(est, x)
    bins = fasthist(rbc, ds)[1]
    return Probabilities(bins)
end

function probabilities_and_outcomes(est::Diversity, x::AbstractVector{T}) where T <: Real
    ds, rbc = similarities_and_binning(est, x)
    return probabilities_and_outcomes(rbc, ds)
end

outcome_space(est::Diversity) = outcome_space(encoding_for_diversity(est.nbins))
total_outcomes(est::Diversity) = est.nbins

function similarities_and_binning(est::Diversity, x::AbstractVector{T}) where T <: Real
    # embed and then calculate cosine similary for each consecutive pair of delay vectors
    τs = 0:est.τ:(est.m - 1)*est.τ
    Y = genembed(x, τs)
    ds = zeros(Float64, length(Y) - 1)
    @inbounds for i in 1:(length(Y)-1)
        ds[i] = cosine_similarity(Y[i], Y[i+1])
    end
    # Cosine similarities are all on [-1.0, 1.0], so just discretize this interval
    rbc = encoding_for_diversity(est.nbins)
    return ds, rbc
end

cosine_similarity(xᵢ, xⱼ) = sum(xᵢ .* xⱼ) / (sqrt(sum(xᵢ .^ 2)) * sqrt(sum(xⱼ .^ 2)))

function encoding_for_diversity(nbins::Int)
    binning = FixedRectangularBinning((range(-1.0, nextfloat(1.0); length = nbins+1),))
    return RectangularBinEncoding(binning)
end
