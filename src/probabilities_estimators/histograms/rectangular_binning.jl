export RectangularBinning
export RectangularBinMapping

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

"""
    RectangularBinMapping <: Encoding

A encoding scheme where the [outcome space](@ref terminology) is a set of
rectangular bins, which identified by their minima and edgelengths.

Used in [`outcomes`](@ref).

    RectangularBinMapping(x, binning::RectangularBinning)

Find the minima along each dimension, and compute appropriate edge lengths for each
dimension of `x`, given a rectangular binning. Put the minima and edgelengths in a
`RectangularBinMapping` that can be then used to map points into bins via
[`outcomes`](@ref).

See also: [`RectangularBinning`](@ref), [`FixedRectangularBinning`](@ref).
"""
struct RectangularBinMapping{M, E} <: Encoding
    binning::RectangularBinning # type specialization isn't useful here; we don't use this.
    mini::M # fields are either static vectors or numbers
    edgelengths::E
end

function RectangularBinMapping(x::AbstractDataset{D,T}, b::RectangularBinning; n_eps = 2) where {D, T}
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

    RectangularBinMapping(b, mini, edgelengths)
end

function RectangularBinMapping(x::AbstractVector{<:Real}, b::RectangularBinning; n_eps = 2)
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

    RectangularBinMapping(b, mini, edgelength)
end

function encode_as_bin(point, b::RectangularBinMapping)
    (; mini, edgelengths) = b
    # Map a data point to its bin edge
    return floor.(Int, (point .- mini) ./ edgelengths)
end

function outcomes(x::Vector_or_Dataset, b::RectangularBinMapping)
    return map(point -> encode_as_bin(point, b), x)
end
