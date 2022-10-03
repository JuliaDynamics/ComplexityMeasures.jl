export RectangularBinning

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
    fasthist(x, binning)[1]
end
function probabilities(x::Vector_or_Dataset, ε::Union{Real, Vector{<:Real}})
    probabilities(x, RectangularBinning(ε))
end


"""
    bin_encoder(x, binning_scheme::RectangularBinning)

Find the minima along each dimension, and compute appropriate
edge lengths for each dimension of `x` given a rectangular binning.
Put them in a `RectangularBinEncoder` that can be then used to map points into bins.
"""
function bin_encoder(x::AbstractDataset{D,T}, b::RectangularBinning) where {D, T}
    # This function always returns static vectors and is type stable
    ϵ = b.ϵ
    mini, maxi = minmaxima(x)
    v = ones(SVector{D,T})
    if ϵ isa Float64 || ϵ isa Vector{<:AbstractFloat}
        edgelengths = ϵ .* v
    elseif ϵ isa Int || ϵ isa Vector{Int}
        edgeslengths_nonadjusted = @. (maxi - mini)/ϵ
        # just taking the next float here is enough to ensure boxes cover data
        edgelengths = nextfloat.(edgeslengths_nonadjusted)
    end
    RectangularBinEncoder(mini, edgelengths)
end

function bin_encoder(x::AbstractVector{<:Real}, b::RectangularBinning)
    # This function always returns numbers and is type stable
    ϵ = b.ϵ
    mini, maxi = extrema(x)
    if ϵ isa AbstractFloat
        edgelength = ϵ
    elseif ϵ isa Int
        edgeslength_nonadjusted = (maxi - mini)/ϵ
        # just taking the next float here should be enough
        edgelength = nextfloat(edgeslength_nonadjusted)
        # edgelengths = ((axismaxima + (edgeslengths_nonadjusted ./ 100)) - axisminima) ./ ϵ
    else
        error("Invalid ϵ for binning of a vector")
    end
    RectangularBinEncoder(mini, edgelength)
end

struct RectangularBinEncoder{M, E} <: AbstractBinEncoder
    mini::M # fields are either static vectors or numbers
    edgelengths::E
end

# This function is the same as `symbolize`.
function encode_as_bins(x::Vector_or_Dataset, b::RectangularBinEncoder)
    (; mini, edgelengths) = b
    # Map each datapoint to its bin edge (hence, we are symbolizing the x here)
    # (notice that this also works for vector x, and broadcasting is ignored)
    bins = map(point -> floor.(Int, (point .- mini) ./ edgelengths), x)
    return bins
end
function symbolize(x::Vector_or_Dataset, b::RectangularBinEncoder)
    return encode_as_bins(x, b)
end
