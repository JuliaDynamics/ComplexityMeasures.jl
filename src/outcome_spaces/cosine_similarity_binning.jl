using DelayEmbeddings

export CosineSimilarityBinning, Diversity

"""
    CosineSimilarityBinning(; m::Int, τ::Int, nbins::Int)

A [`OutcomeSpace`](@ref) based on the cosine similarity [Wang2020](@cite).

It can be used with [`information`](@ref) to compute the "diversity entropy" of an input
timeseries [Wang2020](@cite).

The implementation here allows for `τ != 1`, which was not considered in the original paper.

## Description

CosineSimilarityBinning probabilities are computed as follows.

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

The outcome space for `CosineSimilarityBinning` is the bins of the `[-1, 1]` interval,
and the return configuration is the same as in [`ValueBinning`](@ref) (left bin edge).

## Implements

- [`codify`](@ref). Used for encoding inputs where ordering matters (e.g. time series).
"""
Base.@kwdef struct CosineSimilarityBinning <: CountBasedOutcomeSpace
    m::Int = 2
    τ::Int = 1 # Note: the original paper does not allow τ != 1
    nbins::Int = 5
end

"""
    Diversity

An alias to [`CosineSimilarityBinning`](@ref).
"""
const Diversity = CosineSimilarityBinning

function counts_and_outcomes(o::CosineSimilarityBinning, x::AbstractVector{T}) where T <: Real
    # Cosine similarities are all on [-1.0, 1.0], so just discretize this interval. To
    # do so, we call the `counts_and_outcomes(::RectangularBinEncoding, x)` in the file
    # `encoding_implementations/rectangular_binning.jl`.
    rbc::RectangularBinEncoding = encoding_for_diversity(o.nbins)
    cdists = cosine_similarity_distances(o, x)
    cts, outs = counts_and_outcomes(rbc, cdists)
    return cts, outcomes(cts)
end

function cosine_similarity_distances(o::CosineSimilarityBinning, x::AbstractVector{T}) where T <: Real
    # embed and then calculate cosine similary for each consecutive pair of delay vectors
    τs = 0:o.τ:(o.m - 1)*o.τ
    Y = genembed(x, τs)
    ds = zeros(Float64, length(Y) - 1)
    @inbounds for i in 1:(length(Y)-1)
        # The cosine similarity, by construction, is bounded to [-1, 1]. Due to precision errors,
        # its value may sometimes be slightly outside this range. This causes problems when 
        # computing the histogram over the cosine similarity distances using `RectangularBinEncoding`.
        # By subtracting a machine epsilon, we ensure that  no cosine similarities are encoded 
        # as the outcome `-1`. Note: we could also do this in the construction of the binning
        # by expanding the range slightly. But to keep the binning conceptually aligned with
        # the cosine similarity definition, we apply the correction here.
        ds[i] = min(cosine_similarity(Y[i], Y[i+1]), 1.0 - eps())
    end
    return ds
end

outcome_space(o::CosineSimilarityBinning) = outcome_space(encoding_for_diversity(o.nbins))
total_outcomes(o::CosineSimilarityBinning) = o.nbins

function encoded_space_cardinality(o::CosineSimilarityBinning, x::AbstractVector{<:Real})
    n_pts_embedded = length(x) - (o.m - 1)*o.τ
    # Since we consider cosine similarities for consecutive pairs of embedding points,
    # the last point isn't considered for the histogram.
    return n_pts_embedded - 1
end

cosine_similarity(xᵢ, xⱼ) = cs = sum(xᵢ .* xⱼ) / (sqrt(sum(xᵢ .^ 2)) * sqrt(sum(xⱼ .^ 2)))

function encoding_for_diversity(nbins::Int)
    precise = false
    r = range(-1.0, nextfloat(1.0); length = nbins+1)
    binning = FixedRectangularBinning((r,), precise)
    return RectangularBinEncoding(binning)
end

function codify(o::CosineSimilarityBinning, x::AbstractVector{<:Real})
    τs = 0:o.τ:(o.m - 1)*o.τ
    Y = genembed(x, τs)
    ds = zeros(Float64, length(Y) - 1)
    @inbounds for i in 1:(length(Y)-1)
        ds[i] = cosine_similarity(Y[i], Y[i+1])
    end
    # Cosine similarities are all on [-1.0, 1.0], so just discretize this interval
    rbc = encoding_for_diversity(o.nbins)
    return encode.(Ref(rbc), ds)
end
