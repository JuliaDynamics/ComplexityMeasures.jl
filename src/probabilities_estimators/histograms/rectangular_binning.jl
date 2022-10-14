export RectangularBinning
export RectangularBinEncoder

"""
    RectangularBinning(ϵ) <: AbstractBinning

Rectangular box partition of state space using the scheme `ϵ`.
Binning instructions are deduced from the type of `ϵ` as follows:

1. `ϵ::Int` divides each coordinate axis into `ϵ` equal-length intervals
    that cover all data.
2. `ϵ::Float64` divides each coordinate axis into intervals of fixed size `ϵ`, starting
    from the axis minima until the data is completely covered by boxes.
3. `ϵ::Vector{Int}` divides the i-th coordinate axis into `ϵ[i]` equal-length
    intervals that cover all data.
4. `ϵ::Vector{Float64}` divides the i-th coordinate axis into intervals of fixed size
    `ϵ[i]`, starting from the axis minima until the data is completely covered by boxes.
"""
struct RectangularBinning{E} <: AbstractBinning
    ϵ::E
end

function probabilities(x::Vector_or_Dataset, binning::RectangularBinning)
    fasthist!(x, binning)[1]
end

function probabilities(x::Vector_or_Dataset, ε::Union{Real, Vector{<:Real}})
    probabilities(x, RectangularBinning(ε))
end

# Internal function method extension for `probabilities`
function fasthist!(x::Vector_or_Dataset, ϵ::AbstractBinning)
    encoder = RectangularBinEncoder(x, ϵ)
    bins = symbolize(x, encoder)
    hist = fasthist!(bins)
    return Probabilities(hist), bins, encoder
end

function probabilities_and_events(x, ϵ::RectangularBinning)
    probs, bins, encoder = fasthist!(x, ϵ)
    (; mini, edgelengths) = encoder
    unique!(bins) # `bins` is already sorted from `fasthist!`
    events = map(b -> b .* edgelengths .+ mini, bins)
    return probs, events
end

"""
    RectangularBinEncoder(x, binning::RectangularBinning) <: SymbolizationScheme

Find the minima along each dimension, and compute appropriate
edge lengths for each dimension of `x` given a rectangular binning.
Put them in an `RectangularBinEncoder` that can be then used to map points into bins
via [`symbolize`](@ref).
"""
struct RectangularBinEncoder{M, E} <: SymbolizationScheme
    binning::RectangularBinning # type specialization isn't useful here; we don't use this.
    mini::M # fields are either static vectors or numbers
    edgelengths::E
end

function RectangularBinEncoder(x::AbstractDataset{D,T}, b::RectangularBinning; n_eps = 2) where {D, T}
    # This function always returns static vectors and is type stable
    ϵ = b.ϵ
    mini, maxi = minmaxima(x)
    v = ones(SVector{D,T})
    if ϵ isa Float64 || ϵ isa AbstractVector{<:AbstractFloat}
        edgelengths = ϵ .* v
    elseif ϵ isa Int || ϵ isa Vector{Int}
        edgeslengths_nonadjusted = @. (maxi - mini)/ϵ
        # Just taking nextfloat once here isn't enough for bins to cover data when using
        # `encode_as_bin` later, because subtraction and division leads to loss
        # of precision. We need a slightly bigger number, so apply nextfloat twice.
        edgelengths = nextfloat.(edgeslengths_nonadjusted, n_eps)
    else
        error("Invalid ϵ for binning of a dataset")
    end

    RectangularBinEncoder(b, mini, edgelengths)
end

function RectangularBinEncoder(x::AbstractVector{<:Real}, b::RectangularBinning; n_eps = 2)
    # This function always returns numbers and is type stable
    ϵ = b.ϵ
    mini, maxi = extrema(x)
    if ϵ isa AbstractFloat
        edgelength = ϵ
    elseif ϵ isa Int
        edgeslength_nonadjusted = (maxi - mini)/ϵ
        # Round-off occurs when encoding bins. Applying `nextfloat` twice seems to still
        # ensure that bins cover data. See comment above.
        edgelength = nextfloat(edgeslength_nonadjusted, n_eps)
    else
        error("Invalid ϵ for binning of a vector")
    end

    RectangularBinEncoder(b, mini, edgelength)
end

function encode_as_bin(point, b::RectangularBinEncoder)
    (; mini, edgelengths) = b
    # Map a data point to its bin edge
    return floor.(Int, (point .- mini) ./ edgelengths)
end

function symbolize(x::Vector_or_Dataset, b::RectangularBinEncoder)
    return map(point -> encode_as_bin(point, b), x)
end
