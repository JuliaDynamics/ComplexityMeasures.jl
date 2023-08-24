using DelayEmbeddings

export Diversity

"""
    Diversity(; m::Int, τ::Int, nbins::Int)

A [`OutcomeSpace`](@ref) based on the cosine similarity [Wang2020](@cite).

It can be used with [`information`](@ref) to compute the "diversity entropy" of an input
timeseries [Wang2020](@cite).

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
"""
Base.@kwdef struct Diversity <: CountBasedOutcomeSpace
    m::Int = 2
    τ::Int = 1 # Note: the original paper does not allow τ != 1
    nbins::Int = 5
end

function counts(est::Diversity, x::AbstractVector{T}) where T <: Real
    ds, rbc = similarities_and_binning(est, x)
    bins = fasthist(rbc, ds)[1]
    return bins
end

function counts_and_outcomes(est::Diversity, x::AbstractVector{T}) where T <: Real
    ds, rbc = similarities_and_binning(est, x)
    cts, outcomes = counts_and_outcomes(rbc, ds)
    return cts, outcomes
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

function encoded_space_cardinality(est::Diversity, x::AbstractVector{<:Real})
    n_pts_embedded = length(x) - (est.m - 1)*est.τ
    # Since we consider cosine similarities for consecutive pairs of embedding points,
    # the last point isn't considered for the histogram.
    return n_pts_embedded - 1
end

cosine_similarity(xᵢ, xⱼ) = sum(xᵢ .* xⱼ) / (sqrt(sum(xᵢ .^ 2)) * sqrt(sum(xⱼ .^ 2)))

function encoding_for_diversity(nbins::Int)
    binning = FixedRectangularBinning((range(-1.0, nextfloat(1.0); length = nbins+1),))
    return RectangularBinEncoding(binning)
end
